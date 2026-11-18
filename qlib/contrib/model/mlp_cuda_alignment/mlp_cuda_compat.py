"""TF32 MMA diagnostic primitive. Normal finite operands, no release/performance claim.
Uniform integer exponent alignment inferred from A100 synthetic observations.
"""
import torch

def round_tf32_away(t):
    bits=t.contiguous().view(torch.int32)
    return ((bits+4096)&-8192).view(torch.float32)

def mma4(a,b,c):
    """Four TF32 products plus FP32 accumulator, lane-wise over last dimension."""
    products=a*b
    terms=torch.cat([products,c[None,:]],dim=0).contiguous()
    bits=terms.view(torch.int32).to(torch.int64)
    exp=(bits>>23)&255
    mant=(bits&0x7fffff)|0x800000
    mant=torch.where(exp==0,torch.zeros_like(mant),mant)
    abits=a.contiguous().view(torch.int32).to(torch.int64)
    bbits=b.contiguous().view(torch.int32).to(torch.int64)
    ae=(abits>>23)&255;be=(bbits>>23)&255
    raw_exp=torch.where((ae==0)|(be==0),torch.zeros_like(ae),ae+be-127)
    top=torch.maximum(raw_exp.max(dim=0).values,exp[-1])
    shift=top[None,:]-exp
    extended=mant<<1
    aligned=torch.where(shift>=0,extended>>shift.clamp(0,62),extended<<(-shift).clamp(0,62))
    total=torch.where(bits<0,-aligned,aligned).sum(dim=0)
    mag=total.abs()
    lead=(((mag.to(torch.float32).view(torch.int32).to(torch.int64)>>23)&255)-127).clamp(min=0)
    lead=(lead-(mag<(torch.ones_like(lead)<<lead)).to(torch.int64)).clamp(min=0)
    sh=lead-23
    norm=torch.where(sh>=0,mag>>sh.clamp(min=0),mag<<(-sh).clamp(min=0))
    out_exp=top-24+lead
    out=(out_exp<<23)|(norm&0x7fffff)|torch.where(total<0,torch.full_like(total,0x80000000),torch.zeros_like(total))
    return torch.where(mag==0,torch.zeros_like(out),out).to(torch.int32).view(torch.float32)

def linear_row(x,w,bias):
    """One-row reference prototype; preserve K residue-first traversal."""
    x=round_tf32_away(x);w=round_tf32_away(w)
    K=x.numel();N=w.shape[0];acc=torch.zeros(N,device=x.device)
    residue=K%32 or 32
    for start,end in [(0,residue)]+[(k,k+32) for k in range(residue,K,32)]:
        a=x[start:end,None].expand(-1,N).contiguous();b=w[:,start:end].t().contiguous()
        if len(a)<32:
            pad=torch.zeros((32-len(a),N),device=x.device);a=torch.cat([a,pad]);b=torch.cat([b,pad])
        for k in range(0,32,4):acc=mma4(a[k:k+4],b[k:k+4],acc)
    return acc+bias

def linear_batch(x,w,bias,chunk_rows=2048,tile_k=32):
    """Batched diagnostic prototype; chunking does not change per-row arithmetic."""
    w=round_tf32_away(w);K=w.shape[1];N=w.shape[0];residue=K%tile_k or tile_k;result=[]
    tiles=[(0,residue)]+[(k,k+tile_k) for k in range(residue,K,tile_k)]
    for begin in range(0,len(x),chunk_rows):
        v=round_tf32_away(x[begin:begin+chunk_rows]);B=len(v);acc=torch.zeros(B*N,device=v.device)
        for start,end in tiles:
            for k in range(start,start+tile_k,4):
                valid=max(0,min(4,end-k))
                if valid:
                    aa=v[:,k:k+valid].t().contiguous();bb=w[:,k:k+valid].t().contiguous()
                    if valid<4:
                        aa=torch.cat([aa,torch.zeros((4-valid,B),device=v.device)])
                        bb=torch.cat([bb,torch.zeros((4-valid,N),device=v.device)])
                    a=aa[:,:,None].expand(4,B,N).reshape(4,B*N)
                    b=bb[:,None,:].expand(4,B,N).reshape(4,B*N)
                    acc=mma4(a,b,acc)
        result.append(acc.reshape(B,N)+bias)
    return torch.cat(result,dim=0)

"""Integer FP32 fused multiply-add diagnostic, finite normal operands and zeros.
Keeps exact product bits plus signed sticky information, then rounds once to even.
Subnormal inputs/results and nonfinite operands are outside this prototype's scope.
"""
import torch

def ieee_fma_normal(a,b,c):
    a,b,c=torch.broadcast_tensors(a,b,c)
    av=a.contiguous().view(torch.int32).to(torch.int64)
    bv=b.contiguous().view(torch.int32).to(torch.int64)
    cv=c.contiguous().view(torch.int32).to(torch.int64)
    ae=(av>>23)&255;be=(bv>>23)&255;ce=(cv>>23)&255
    am=torch.where(ae==0,torch.zeros_like(ae),(av&0x7fffff)|0x800000)
    bm=torch.where(be==0,torch.zeros_like(be),(bv&0x7fffff)|0x800000)
    cm=torch.where(ce==0,torch.zeros_like(ce),(cv&0x7fffff)|0x800000)
    pm=am*bm;psign=torch.where((av^bv)<0,-torch.ones_like(pm),torch.ones_like(pm))
    csign=torch.where(cv<0,-torch.ones_like(cm),torch.ones_like(cm))
    pe=ae+be-254-46;cs=ce-127-23
    ptop=torch.where(pm==0,torch.full_like(pm,-10000),ae+be-254+1)
    ctop=torch.where(cm==0,torch.full_like(cm,-10000),ce-127)
    scale=torch.maximum(ptop,ctop)-60
    scale=torch.where((pm==0)&(cm==0),torch.zeros_like(scale),scale)
    def align(m,e):
        sh=scale-e;right=sh.clamp(0,62);left=(-sh).clamp(0,62)
        aligned=torch.where(sh>=0,m>>right,m<<left)
        mask=(torch.ones_like(right)<<right)-1
        lost=(sh>0)&((m&mask)!=0)
        return aligned,lost
    pi,plost=align(pm,pe);ci,clost=align(cm,cs)
    total=psign*pi+csign*ci
    residual=psign*plost.to(torch.int64)+csign*clost.to(torch.int64)
    sign=torch.where(total<0,-torch.ones_like(total),torch.ones_like(total))
    sticky=residual*sign;mag=total.abs()
    lead=(((mag.to(torch.float32).view(torch.int32).to(torch.int64)>>23)&255)-127).clamp(min=0)
    lead=(lead-(mag<(torch.ones_like(lead)<<lead)).to(torch.int64)).clamp(min=0)
    sh=lead-23;right=sh.clamp(min=0)
    mant=torch.where(sh>=0,mag>>right,mag<<(-sh).clamp(min=0))
    mask=(torch.ones_like(right)<<right)-1;low=mag&mask
    half=torch.ones_like(right)<<(right-1).clamp(min=0)
    up=(sh>0)&((low>half)|((low==half)&((sticky>0)|((sticky==0)&((mant&1)!=0)))))
    mant=mant+up.to(torch.int64);carry=mant>=0x1000000
    mant=torch.where(carry,mant>>1,mant)
    exponent=scale+lead+127+carry.to(torch.int64)
    bits=(exponent<<23)|(mant&0x7fffff)|torch.where(total<0,torch.full_like(total,0x80000000),torch.zeros_like(total))
    bits=torch.where(mag==0,torch.zeros_like(bits),bits)
    return bits.to(torch.int32).view(torch.float32)

"""NPU FP32 compensated arithmetic for inference BN coefficient compatibility."""
import torch

def two_sum(a,b):
    s=a+b;bb=s-a
    return s,(a-(s-bb))+(b-bb)

def two_product(a,b):
    p=a*b;ca=a*4097.0;ah=ca-(ca-a);al=a-ah;cb=b*4097.0;bh=cb-(cb-b);bl=b-bh
    return p,((ah*bh-p)+ah*bl+al*bh)+al*bl


def diagnostic_fma(a,b,c):
 return ieee_fma_normal(a,b,c)

def gemv_16lane(x,w,bias):
 if x.shape[1]!=256:raise ValueError('Frozen model GEMV diagnostic requires K=256')
 acc=torch.zeros((len(x),16),dtype=x.dtype,device=x.device)
 weight=w.reshape(-1)
 for j in range(16):
  start=j*16
  acc=diagnostic_fma(x[:,start:start+16],weight[start:start+16],acc)
 y=acc[:,0]+acc[:,1]
 for lane in range(2,16):y=y+acc[:,lane]
 return (y+bias.reshape(-1)[0])[:,None]

# Training-forward compatibility probe only. Backward remains native NPU math
# and has NOT been accepted as CUDA numerically compatible.
"""CUDA Reduce.cuh sum(dim=0), contiguous FP32 N8192,C256, vec4.
Source-derived block=(32,4), grid=(2,128), vt=4. Not a general reduction.
"""
import torch

def cuda_sum_channels_8192(g):
 if tuple(g.shape)!=(8192,256):raise ValueError('Unprofiled reduction shape')
 data=g.reshape(4,4,128,4,256)
 partial=torch.zeros((4,128,4,256),device=g.device)
 for i in range(4):partial=partial+data[i]
 a=partial[0]
 for j in range(1,4):a=a+partial[j]
 for off in [2,1]:a=a[:,:off]+a[:,off:2*off]
 staged=a[:,0].reshape(32,4,256)
 a=torch.zeros((4,256),device=g.device)
 for j in range(32):a=a+staged[j]
 for off in [2,1]:a=a[:off]+a[off:2*off]
 return a[0]

"""Verified first-layer weight gradient diagnostic, real batch shape only.
CUDA Params: K8192, split36, K228; residue-first MMA and serial split sum.
"""
def first_weight_splitk_228(g,x):
 if tuple(g.shape)!=(8192,256) or tuple(x.shape)!=(8192,157):
  raise ValueError('Unprofiled first-weight gradient shape')
 gs=[];xs=[]
 for split in range(36):
  start=split*228;end=min((split+1)*228,8192)
  gg=g[start:end];xx=x[start:end]
  residue=len(gg)%32 or 32
  def pack(v):
   v=torch.cat([v[:residue],torch.zeros((32-residue,v.shape[1]),device=v.device),v[residue:]])
   return torch.cat([v,torch.zeros((256-len(v),v.shape[1]),device=v.device)])
  gs.append(pack(gg));xs.append(pack(xx))
 g=round_tf32_away(torch.stack(gs));x=round_tf32_away(torch.stack(xs));parts=[]
 with torch.no_grad():
  for start in range(0,36,4):
   aa=g[start:start+4];bb=x[start:start+4];acc=torch.zeros(4*256*157,device=g.device)
   for k in range(0,256,4):
    a=aa[:,k:k+4].permute(1,0,2)[:,:,:,None].expand(4,4,256,157).reshape(4,-1)
    b=bb[:,k:k+4].permute(1,0,2)[:,:,None,:].expand(4,4,256,157).reshape(4,-1)
    acc=mma4(a,b,acc)
   parts.append(acc.reshape(4,256,157))
  partial=torch.cat(parts);result=torch.zeros_like(partial[0])
  for i in range(36):result=result+partial[i]
 return result

"""Verified diagnostic schedule for one CUDA GEMVX shape: K8192,N256, split9.
Each split starts at 911*s; last904. Sixteen lanes, serial per-lane FMA,
descending binary-tree lane reduction, then serial sum of the nine splits.
Only the controlled real first batch has passed complete CUDA gradient hashes.
"""
def last_weight_gemvx_911(g,x):
 if tuple(g.shape)!=(8192,1) or tuple(x.shape)!=(8192,256):
  raise ValueError('Unprofiled last-weight GEMVX shape')
 xs=[];gs=[];g=g.reshape(-1)
 for split in range(9):
  start=split*911;end=min(start+911,8192)
  xs.append(torch.cat([x[start:end],torch.zeros((912-(end-start),256),device=x.device)]))
  gs.append(torch.cat([g[start:end],torch.zeros(912-(end-start),device=x.device)]))
 data=torch.stack(xs).reshape(9,57,16,256)
 weights=torch.stack(gs).reshape(9,57,16,1)
 acc=torch.zeros((9,16,256),device=x.device)
 for j in range(57):acc=ieee_fma_normal(data[:,j],weights[:,j],acc)
 for off in [8,4,2,1]:acc=acc[:,:off]+acc[:,off:2*off]
 total=torch.zeros(256,device=x.device)
 for split in range(9):total=total+acc[split,0]
 return total[None,:]


def cuda_sum_flat_8192(g):
 if g.numel()!=8192 or not g.is_contiguous():raise ValueError('Unprofiled flat reduction')
 values=g.reshape(4,512,4)
 acc=torch.zeros((512,4),device=g.device)
 for j in range(4):acc=acc+values[j]
 total=acc[:,0]+acc[:,1]
 total=total+acc[:,2];total=total+acc[:,3]
 for off in [256,128,64,32,16,8,4,2,1]:total=total[:off]+total[off:2*off]
 return total.reshape(1)

class ForwardReplayLinear(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,w,b):
  ctx.save_for_backward(x,w)
  if w.shape==(256,157):return linear_batch(x,w,b,tile_k=32)
  if w.shape==(1,256):return gemv_16lane(x,w,b)
  raise ValueError('Unprofiled training-forward linear shape')
 @staticmethod
 def backward(ctx,g):
  x,w=ctx.saved_tensors
  gx=g.matmul(w) if ctx.needs_input_grad[0] else None
  gw=(first_weight_splitk_228(g,x) if tuple(w.shape)==(256,157) else last_weight_gemvx_911(g,x)) if ctx.needs_input_grad[1] else None
  gb=(cuda_sum_channels_8192(g) if tuple(g.shape)==(8192,256) else cuda_sum_flat_8192(g)) if ctx.needs_input_grad[2] else None
  return gx,gw,gb
"""NPU integer port of generic SFU-Sim RSQ model, not model-specific coefficients.
Derived from MIT SFU-Sim commit 42926e7c95e210e21b2a5a624368d60b24cbabfa.
Source: https://github.com/hzX40000000/SFU-Sim
All per-input arithmetic runs on NPU. This models bare MUFU.RSQ/rsqrt.approx.ftz.f32.
"""
# MIT License
#
# Copyright (c) 2026 SFU-Sim contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
_TABLES={'even': {'a2': [3016, 2900, 2792, 2688, 2592, 2500, 2412, 2328, 2248, 2172, 2104, 2032, 1968, 1904, 1844, 1784, 1732, 1680, 1628, 1580, 1532, 1488, 1448, 1408, 1368, 1328, 1292, 1256, 1220, 1192, 1160, 1132, 1100, 1072, 1048, 1020, 992, 968, 944, 924, 900, 880, 860, 840, 820, 804, 784, 764, 748, 732, 716, 704, 688, 672, 660, 648, 632, 616, 604, 592, 580, 572, 560, 548], 'a1': [-65531, -64024, -62575, -61179, -59835, -58539, -57289, -56083, -54919, -53794, -52708, -51657, -50641, -49658, -48706, -47784, -46891, -46026, -45186, -44372, -43582, -42815, -42071, -41348, -40645, -39962, -39298, -38652, -38023, -37412, -36816, -36237, -35672, -35122, -34586, -34063, -33553, -33056, -32571, -32098, -31636, -31185, -30745, -30315, -29895, -29485, -29084, -28691, -28308, -27933, -27566, -27208, -26857, -26513, -26177, -25848, -25525, -25209, -24900, -24597, -24300, -24009, -23724, -23444], 'a0': [1099511611390, 1091021029372, 1082724204543, 1074613796863, 1066682974206, 1058925166591, 1051334197247, 1043904184319, 1036629524479, 1029504827390, 1022525014012, 1015685332991, 1008981049343, 1002407854078, 995961421822, 989637804024, 983433084927, 977343692798, 971365974014, 965496651772, 959732449278, 954070253567, 948507099135, 943040151548, 937666658303, 932384030718, 927189647358, 922081132540, 917056126975, 912112386044, 907247714296, 902460112895, 897747517436, 893107978238, 888539578366, 884040630271, 879609331710, 875243995128, 870943031295, 866704801790, 862527913980, 858410778623, 854352084991, 850350424063, 846404468735, 842512924671, 838674612220, 834888237055, 831152717820, 827466858495, 823829610494, 820239925240, 816696770557, 813199114239, 809746022399, 806336528376, 802969780223, 799644860415, 796360900604, 793117048831, 789912518652, 786746474495, 783618277372, 780527058940]}, 'odd': {'a2': [2132, 2052, 1972, 1900, 1836, 1768, 1708, 1648, 1592, 1536, 1484, 1436, 1392, 1344, 1304, 1264, 1224, 1188, 1152, 1116, 1084, 1052, 1024, 992, 964, 940, 916, 888, 868, 840, 820, 796, 780, 760, 740, 720, 704, 684, 668, 652, 636, 620, 608, 592, 580, 568, 552, 544, 532, 520, 504, 496, 488, 476, 468, 456, 448, 440, 428, 420, 412, 404, 396, 388], 'a1': [-46337, -45272, -44247, -43260, -42310, -41393, -40510, -39657, -38834, -38038, -37270, -36527, -35809, -35113, -34440, -33789, -33157, -32545, -31952, -31376, -30817, -30275, -29749, -29237, -28740, -28257, -27788, -27331, -26887, -26454, -26033, -25623, -25224, -24835, -24456, -24086, -23726, -23374, -23031, -22697, -22370, -22051, -21740, -21436, -21139, -20849, -20565, -20288, -20017, -19752, -19492, -19239, -18991, -18748, -18510, -18277, -18049, -17826, -17607, -17393, -17183, -16977, -16775, -16578], 'a0': [777472098303, 771468361724, 765601644543, 759866703868, 754258739198, 748773146623, 743405567998, 738151727103, 733007773694, 727969841150, 723034423294, 718197997567, 713457385471, 708809392127, 704251052024, 699779629055, 695392223231, 691086344191, 686859517936, 682709270524, 678633324543, 674629566463, 670695833599, 666830127103, 663030480892, 659295043583, 655622062076, 652009832447, 648456585215, 644960878590, 641521041407, 638135713791, 634803339260, 631522689023, 628292403196, 625111138302, 621977731070, 618890969086, 615849721855, 612852891647, 609899347966, 606988124159, 604118171644, 601288589308, 598498344959, 595746619391, 593032527871, 590355120096, 587713691647, 585107423224, 582535544828, 579997253631, 577491828735, 575018647548, 572576890878, 570166050815, 567785373695, 565434286078, 563112198143, 560818454526, 558552514559, 556313788415, 554101784575, 551916027902]}}
_CACHE={}

def _constants(device):
 key=str(device)
 if key not in _CACHE:
  coeff={parity:{name:torch.tensor(v,dtype=torch.int64,device=device) for name,v in row.items()} for parity,row in _TABLES.items()}
  u=torch.arange(32768,dtype=torch.int64,device=device);total=torch.zeros_like(u)
  for i in range(15):
   high=u & ~((1<<(14-i))-1)
   total=total+(((u>>i)&1)*(high<<i))
  squares=(total>>15)&65535
  _CACHE[key]=(coeff,squares)
 return _CACHE[key]

def rsqrt_cuda_ftz(x):
 if x.dtype!=torch.float32 or x.device.type!='npu':raise ValueError('Requires FP32 NPU input')
 coeff,squares=_constants(x.device);bits=x.contiguous().view(torch.int32).to(torch.int64)
 ab=bits&0x7fffffff;exp=(bits>>23)&255;E=exp-127;m=bits&0x7fffff;odd=(E&1)!=0
 index=(m>>17).reshape(-1);r=m&0x1ffff
 def gather(name):
  even=coeff['even'][name][index].reshape(x.shape);uneven=coeff['odd'][name][index].reshape(x.shape)
  return torch.where(odd,uneven,even)
 square=squares[(r>>2).reshape(-1)].reshape(x.shape)
 q=((gather('a0')+gather('a1')*r+gather('a2')*square)>>16)&0xffffffff
 lead=(((q.to(torch.float32).view(torch.int32).to(torch.int64)>>23)&255)-127).clamp(min=0)
 lead=(lead-(q<(torch.ones_like(lead)<<lead)).to(torch.int64)).clamp(min=0)
 shift=lead-23;right=shift.clamp(min=0)
 mant=torch.where(shift>0,q>>right,q<<(-shift).clamp(min=0))
 remainder=q&((torch.ones_like(right)<<right)-1);half=torch.ones_like(right)<<(right-1).clamp(min=0)
 up=(shift>0)&((remainder>half)|((remainder==half)&((mant&1)!=0)))
 mant=mant+up.to(torch.int64);carry=mant==0x1000000;scale=-(E>>1)
 exponent=scale+lead-24+127+carry.to(torch.int64)
 result=(exponent<<23)|torch.where(carry,torch.zeros_like(mant),mant&0x7fffff)
 result=torch.where((m==0)&~odd,(scale+127)<<23,result)
 result=torch.where(bits<0,torch.full_like(result,0x7fffffff),result)
 result=torch.where(exp==0,(bits&0x80000000)|0x7f800000,result)
 result=torch.where(ab==0x7f800000,torch.where(bits<0,torch.full_like(result,0x7fffffff),torch.zeros_like(result)),result)
 result=torch.where(ab>0x7f800000,torch.full_like(result,0x7fffffff),result)
 return result.to(torch.int32).view(torch.float32)

"""Source-guided CUDA channels-last Welford replay, N=8192 C=256 only.
Reference: pytorch v2.10.0 Normalization.cuh flexible_launch_configs,
batch_norm_collect_statistics_channels_last_kernel and welford_merge_element.
This is an unaccepted diagnostic, not a general training implementation.
"""
import torch

def welford_stats_8192(x,fma,mean_order='left',local_m2='rounded'):
 if tuple(x.shape)!=(8192,256):raise ValueError('Only profiled diagnostic shape supported')
 # CUDA block=(32,16), grid=(8,32); four concurrent per-thread streams.
 data=x.reshape(4,4,32,16,256)
 means=torch.zeros((4,32,16,256),device=x.device);m2=torch.zeros_like(means)
 for i in range(4):
  delta=data[i]-means
  means=fma(delta,torch.full_like(delta,1.0/(i+1)),means)
  delta1=data[i]-means
  m2=fma(delta,delta1,m2) if local_m2=='fused' else m2+delta*delta1
 def merge(m,v,c,mn,vn,cn):
  factor=1.0/(c+cn);d=m-mn
  if mean_order=='left':weighted=fma(mn,torch.full_like(mn,float(cn)),m*c)
  else:weighted=fma(m,torch.full_like(m,float(c)),mn*cn)
  out_m=weighted*factor
  correction=((d*d)*cn)*c
  out_v=v+fma(correction,torch.full_like(correction,factor),vn)
  return out_m,out_v,c+cn
 m=means[0];v=m2[0];count=4
 for j in range(1,4):m,v,count=merge(m,v,count,means[j],m2[j],4)
 # Vertical shared-memory tree, offsets 8,4,2,1.
 for offset in [8,4,2,1]:m,v,count=merge(m[:,:offset],v[:,:offset],count,m[:,offset:2*offset],v[:,offset:2*offset],count)
 m=m[:,0];v=v[:,0];count=256
 # Last CUDA block: each y-thread combines staging blocks y and y+16.
 m,v,count=merge(m[:16],v[:16],count,m[16:],v[16:],count)
 for offset in [8,4,2,1]:m,v,count=merge(m[:offset],v[:offset],count,m[offset:2*offset],v[offset:2*offset],count)
 return m[0],v[0]/count

"""Source-guided CUDA BN backward replay for N8192,C256 only; unaccepted diagnostic.
Source: PyTorch v2.10.0 Normalization.cuh, channels-last backward reduce/elemt.
CUDA contraction choice uses explicit FMA and must be verified against observed hashes.
"""
import torch

def bn_backward_8192(g,x,w,mean,inv,fma):
 if tuple(x.shape)!=(8192,256) or g.shape!=x.shape:raise ValueError('Unprofiled BN backward shape')
 data=x.reshape(4,4,32,16,256);go=g.reshape(4,4,32,16,256)
 sy=torch.zeros((4,32,16,256),device=x.device);sx=torch.zeros_like(sy)
 for i in range(4):
  sy=sy+go[i]
  sx=fma(go[i],data[i]-mean,sx)
 a=sy[0];b=sx[0]
 for j in range(1,4):a=a+sy[j];b=b+sx[j]
 for off in [8,4,2,1]:a=a[:,:off]+a[:,off:2*off];b=b[:,:off]+b[:,off:2*off]
 a=a[:,0];b=b[:,0]
 a=a[:16]+a[16:];b=b[:16]+b[16:]
 for off in [8,4,2,1]:a=a[:off]+a[off:2*off];b=b[:off]+b[off:2*off]
 a=a[0];b=b[0]
 factor=((inv*inv)*b)*(1.0/8192)
 gx=fma(-(x-mean),factor,g-a*(1.0/8192))*(w*inv)
 return gx,b*inv,a

class ReplayBN(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,w,b,rm,rv):
  mean,var=welford_stats_8192(x,ieee_fma_normal,'left','rounded')
  inv=rsqrt_cuda_ftz(var+torch.full_like(var,1e-5))
  ctx.save_for_backward(x,w,mean,inv)
  mom=torch.full_like(mean,.1);other=1-mom
  rm.copy_(ieee_fma_normal(mean,mom,rm*other))
  rv.copy_(ieee_fma_normal(var*(8192/8191),mom,rv*other))
  return ieee_fma_normal((x-mean)*w,inv,b)
 @staticmethod
 def backward(ctx,g):
  x,w,mean,inv=ctx.saved_tensors
  gx,gw,gb=bn_backward_8192(g,x,w,mean,inv,ieee_fma_normal)
  return gx,gw,gb,None,None
original_bn_forward=torch.nn.BatchNorm1d.forward
def replay_bn_forward(mod,x):
 if not mod.training:return original_bn_forward(mod,x)
 if tuple(x.shape)!=(8192,256):raise ValueError('Unprofiled BN shape')
 mod.num_batches_tracked.add_(1)
 return ReplayBN.apply(x,mod.weight,mod.bias,mod.running_mean,mod.running_var)


class CudaMean8192(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x):
  if x.numel()!=8192 or not x.is_contiguous():raise ValueError('Unprofiled reduction shape')
  # Reduce.cuh input_vec_size=4, block=(512,1), no inter-CTA reduction.
  v=x.reshape(4,512,4)
  acc=torch.zeros((512,4),device=x.device)
  for i in range(4):acc=acc+v[i]
  total=acc[:,0]+acc[:,1]
  total=total+acc[:,2];total=total+acc[:,3]
  for offset in [256,128,64,32,16,8,4,2,1]:total=total[:offset]+total[offset:2*offset]
  ctx.shape=x.shape
  return total[0]*(1.0/8192)
 @staticmethod
 def backward(ctx,g):return g.expand(ctx.shape)/8192

"""NPU integer FP32 divide, round nearest even. Finite normal inputs/results or zero.
No FP64 or CPU arithmetic fallback. Subnormal/nonfinite cases are unvalidated.
"""
import torch

def ieee_div_normal(a,b):
 a,b=torch.broadcast_tensors(a,b)
 av=a.contiguous().view(torch.int32).to(torch.int64);bv=b.contiguous().view(torch.int32).to(torch.int64)
 ae=(av>>23)&255;be=(bv>>23)&255
 am=(av&0x7fffff)|0x800000;bm=(bv&0x7fffff)|0x800000
 lower=am<bm;sh=23+lower.to(torch.int64);num=am<<sh
 q=torch.zeros_like(num);rem=num
 for bit in range(24,-1,-1):
  trial=bm<<bit;take=rem>=trial
  rem=torch.where(take,rem-trial,rem);q=q|(take.to(torch.int64)<<bit)
 q=q+((rem*2>bm)|((rem*2==bm)&((q&1)!=0))).to(torch.int64)
 carry=q>=0x1000000;q=torch.where(carry,q>>1,q)
 exp=ae-be+127-lower.to(torch.int64)+carry.to(torch.int64)
 sign=(av^bv)&0x80000000
 bits=sign|(exp<<23)|(q&0x7fffff)
 bits=torch.where(ae==0,sign,bits)
 return bits.to(torch.int32).view(torch.float32)

original_foreach_sqrt=torch._foreach_sqrt
def replay_foreach_sqrt(tensors):return [torch.sqrt(t) for t in tensors]
original_foreach_div=torch._foreach_div_
original_foreach_addcdiv=torch._foreach_addcdiv_
def replay_foreach_div(tensors,scalars):
 for i,t in enumerate(tensors):
  scalar=scalars[i] if isinstance(scalars,(list,tuple)) else scalars
  t.copy_(ieee_div_normal(t,torch.full_like(t,scalar)))
 return tensors
def replay_foreach_addcdiv(tensors,numerators,denominators,scalars=1):
 for i,t in enumerate(tensors):
  scalar=scalars[i] if isinstance(scalars,(list,tuple)) else scalars
  quotient=ieee_div_normal(numerators[i],denominators[i])
  t.copy_(ieee_fma_normal(quotient,torch.full_like(t,scalar),t))
 return tensors

original_linear_forward=torch.nn.Linear.forward
def training_forward_linear(mod,x):
 if (mod.in_features,mod.out_features) in [(157,256),(256,1)]:return ForwardReplayLinear.apply(x,mod.weight,mod.bias)
 return original_linear_forward(mod,x)


"""CUDA-compatible Philox4x32-10 candidate on NPU integer tensors.
Sources: PyTorch v2.10.0 PhiloxRNGEngine.h and native/cuda/Dropout.cu.
Uniform conversion and actual CUDA mask stream remain subject to A100 v42 checks.
"""
import torch

def philox4x32(seed,subsequence,counter):
 mask=0xffffffff
 c0=counter&mask;c1=(counter>>32)&mask
 c2=subsequence&mask;c3=(subsequence>>32)&mask
 k0=seed&mask;k1=(seed>>32)&mask
 for _ in range(10):
  a=c0*0xD2511F53;b=c2*0xCD9E8D57
  c0,c1,c2,c3=((b>>32)&mask)^c1^k0,b&mask,((a>>32)&mask)^c3^k1,a&mask
  k0=(k0+0x9E3779B9)&mask;k1=(k1+0xBB67AE85)&mask
 return torch.stack([c0,c1,c2,c3],dim=-1)

def native_cuda_mask_candidate(shape,seed,offset,sms=108):
 n=1
 for size in shape:n*=size
 if n%4 or offset%4:raise ValueError('Only aligned vec4 mask diagnostic supported')
 threads=min((n+255)//256,sms*8)*256
 group=torch.arange(n//4,device='npu:0',dtype=torch.int64)
 sub=group%threads;counter=group//threads+offset//4
 bits=philox4x32(seed,sub,counter)
 uniform=bits.to(torch.float32)*(2.0**-32)+(2.0**-33)
 keep=(uniform<torch.full_like(uniform,.95)).reshape(shape)
 advance=((n-1)//(threads*4)+1)*4
 return keep,offset+advance

class NativeCudaDropoutReplay(torch.autograd.Function):
 @staticmethod
 def forward(ctx,x,mask,scale):
  ctx.save_for_backward(mask);ctx.scale=scale
  return (x*mask.to(x.dtype))*scale
 @staticmethod
 def backward(ctx,g):
  mask,=ctx.saved_tensors
  return (g*mask.to(g.dtype))*ctx.scale,None,None
native_rng_offset=0


"""CUDA Reduce.cuh vec4 single-block FP32 mean of nonnegative loss values.
Source-derived block512 and sequential four accumulators, descending tree.
Restricted to contiguous n in [128,131071]; actual shape must be verified.
"""
import torch

def cuda_mean_singleblock_positive(values):
 n=values.numel()
 if n<128 or n>=131072:raise ValueError('Unprofiled CUDA mean launch regime')
 v=values.contiguous().reshape(-1);vectors=v[:n//4*4].reshape(-1,4)
 acc=torch.zeros((512,4),device=v.device)
 for start in range(0,len(vectors),512):
  chunk=vectors[start:start+512]
  if len(chunk)<512:chunk=torch.cat([chunk,torch.zeros((512-len(chunk),4),device=v.device)])
  acc=acc+chunk
 if n%4:
  acc=acc.clone();acc[:n%4,0]=acc[:n%4,0]+v[n//4*4:]
 total=acc[:,0]+acc[:,1];total=total+acc[:,2];total=total+acc[:,3]
 for off in [256,128,64,32,16,8,4,2,1]:total=total[:off]+total[off:2*off]
 return total[0]*(1.0/n)

from contextlib import AbstractContextManager
import json,time,hashlib
import numpy as np
class CompatibilityContext(AbstractContextManager):
 """Isolated Alpha158 MLP arithmetic replay; no formal installation/performance claim."""
 def __init__(self,seed=20261116,emit=None):
  self.seed=seed;self.offset=0;self.step=0;self.emit=emit or (lambda record:None)
  self._saved=[];self.last_loss=None;self.model=None;self.indices=None;self.raw_x=None;self.pred_hash=None
 def _patch(self,owner,key,value):
  existed=not isinstance(owner,type) or key in owner.__dict__
  self._saved.append((owner,key,getattr(owner,key),existed));setattr(owner,key,value)
 def __enter__(self):
  ctx=self
  old_linear=torch.nn.Linear.forward;old_bn=torch.nn.BatchNorm1d.forward
  old_drop=torch.nn.Dropout.forward;old_mean=torch.Tensor.mean
  def linear(mod,x):
   if x.device.type!='npu':return old_linear(mod,x)
   if (mod.in_features,mod.out_features)==(157,256):
    if mod.training:return ForwardReplayLinear.apply(x,mod.weight,mod.bias)
    tiles={8096:32,3705:16,2135:16}
    if len(x) not in tiles:raise ValueError('Unverified evaluation shape '+str(tuple(x.shape)))
    return linear_batch(x,mod.weight,mod.bias,tile_k=tiles[len(x)])
   if (mod.in_features,mod.out_features)==(256,1):
    return ForwardReplayLinear.apply(x,mod.weight,mod.bias) if mod.training else gemv_16lane(x,mod.weight,mod.bias)
   raise ValueError('Unverified Linear configuration')
  def bn(mod,x):
   if x.device.type!='npu':return old_bn(mod,x)
   if mod.training:return replay_bn_forward(mod,x)
   key=(id(mod.running_var),mod.running_var._version,str(mod.running_var.device))
   if getattr(mod,'_compat_rsq_key',None)!=key:
    mod._compat_rsq=rsqrt_cuda_ftz(mod.running_var+torch.full_like(mod.running_var,mod.eps))
    mod._compat_rsq_key=key
   return ieee_fma_normal((x-mod.running_mean)*mod.weight,mod._compat_rsq,mod.bias)
  def dropout(mod,x):
   if x.device.type!='npu':return old_drop(mod,x)
   if not mod.training:return x
   if mod.p!=.05:raise ValueError('Unverified dropout probability')
   if ctx.step<3 and x.shape[1]==157:ctx.raw_x=x.detach()
   mask,ctx.offset=native_cuda_mask_candidate(tuple(x.shape),ctx.seed,ctx.offset,108)
   return NativeCudaDropoutReplay.apply(x,mask,1/(1-mod.p))
  def mean(t,*args,**kw):
   if t.device.type=='npu' and t.dtype==torch.float32 and not args and not kw:
    if t.numel()==8192:return CudaMean8192.apply(t)
    if t.numel()==125145:return cuda_mean_singleblock_positive(t)
   return old_mean(t,*args,**kw)
  self._patch(torch.nn.Linear,'forward',linear)
  self._patch(torch.nn.BatchNorm1d,'forward',bn)
  self._patch(torch.nn.Dropout,'forward',dropout)
  self._patch(torch.Tensor,'mean',mean)
  self._patch(torch,'_foreach_sqrt',replay_foreach_sqrt)
  self._patch(torch,'_foreach_div_',replay_foreach_div)
  self._patch(torch,'_foreach_addcdiv_',replay_foreach_addcdiv)
  self._patch(torch.npu.matmul,'allow_hf32',False)
  from qlib.contrib.model.pytorch_nn import DNNModelPytorch
  old_loss=DNNModelPytorch.get_loss;old_metric=DNNModelPytorch.get_metric
  old_init=DNNModelPytorch.__init__;old_choice=np.random.choice
  old_setstate=DNNModelPytorch.__setstate__
  def setstate(model,state):
   old_setstate(model,state)
   model._infer_cache_src=None;model._infer_cache_tensor=None
   model.dnn_model.to(model.device)
  def aggregate(items):
   hashes={k:hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest() for k,t in items}
   return hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()
  def init(model,*args,**kw):
   old_init(model,*args,**kw);ctx.model=model
   if model.seed!=ctx.seed:raise ValueError('Compatibility seed must match configured model seed')
   ctx.emit({'event':'initial','state_hash':aggregate(model.dnn_model.state_dict().items())})
  def choice(*args,**kw):
   value=old_choice(*args,**kw)
   if args and isinstance(args[0],(int,np.integer)) and args[0]==451180 and np.asarray(value).size==8192:
    ctx.indices=np.asarray(value,dtype='<i8').copy()
   return value
  old_step=torch.optim.Adam.step
  def loss(model,*args,**kw):
   value=old_loss(model,*args,**kw)
   if model.dnn_model.training:
    ctx.last_loss=value.detach()
    if ctx.step<3:
     ctx.pred_hash=aggregate([('pred',args[0])])
     if ctx.indices is not None and ctx.raw_x is not None:
      raw=ctx.raw_x.cpu().numpy().tobytes()+args[2].detach().cpu().numpy().tobytes()+ctx.indices.tobytes()
      ctx.emit({'event':'batch','step':ctx.step+1,'input_hash':hashlib.sha256(raw).hexdigest()})
   else:ctx.emit({'event':'validation_loss','step':ctx.step,'loss':value.item()})
   return value
  def metric(model,*args,**kw):
   pred=args[0] if args else kw['pred']
   if pred.device.type=='npu' and pred.numel()==125145:
    from .cuda_validation_ic import validation_ic
    target=args[1] if len(args)>1 else kw['target']
    index=args[2] if len(args)>2 else kw['index']
    dates=index.get_level_values(0)
    starts=np.r_[0,np.flatnonzero(dates[1:]!=dates[:-1])+1,len(index)]
    value=validation_ic(pred.reshape(-1),target.reshape(-1),starts)
   else:value=old_metric(model,*args,**kw)
   ctx.emit({'event':'validation_metric','step':ctx.step,'metric':value.item()})
   return value
  def step(opt,*args,**kw):
   value=old_step(opt,*args,**kw);ctx.step+=1
   row={'event':'train_step','step':ctx.step,'loss':ctx.last_loss.item(),'rng_offset':ctx.offset}
   if ctx.step<=3:
    model=ctx.model;net=model.dnn_model
    row['pred_hash']=ctx.pred_hash
    row['grad_hash']=aggregate([(n,t.grad) for n,t in net.named_parameters()])
    row['state_hash']=aggregate(net.state_dict().items())
    row['adam_hash']=aggregate([(n+'/'+k,v) for n,t in net.named_parameters() for k,v in opt.state[t].items() if isinstance(v,torch.Tensor)])
   ctx.emit(row)
   return value
  self._patch(DNNModelPytorch,'__init__',init)
  self._patch(DNNModelPytorch,'__setstate__',setstate)
  self._patch(np.random,'choice',choice)
  self._patch(DNNModelPytorch,'get_loss',loss)
  self._patch(DNNModelPytorch,'get_metric',metric)
  self._patch(torch.optim.Adam,'step',step)
  return self
 def __exit__(self,kind,value,tb):
  for owner,key,old,existed in reversed(self._saved):
   if existed:setattr(owner,key,old)
   else:delattr(owner,key)
  self._saved=[]
  return False
