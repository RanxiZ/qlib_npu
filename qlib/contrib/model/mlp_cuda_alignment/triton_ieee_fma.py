"""Experimental Ascend Triton fusion of the verified finite-normal FP32 FMA.
No release/performance acceptance until complete arithmetic regression passes.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _align(m,e,scale):
 sh=scale-e
 right=tl.minimum(tl.maximum(sh,0),62)
 left=tl.minimum(tl.maximum(-sh,0),62)
 aligned=tl.where(sh>=0,m>>right,m<<left)
 one=tl.full((),1,tl.int64)
 lost=(sh>0)&((m&((one<<right)-1))!=0)
 return aligned,lost

@triton.jit
def _fma_kernel(A,B,C,O,N,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 av=tl.load(A+i,i<N,other=0).to(tl.int32,bitcast=True).to(tl.int64)
 bv=tl.load(B+i,i<N,other=0).to(tl.int32,bitcast=True).to(tl.int64)
 cv=tl.load(C+i,i<N,other=0).to(tl.int32,bitcast=True).to(tl.int64)
 ae=(av>>23)&255;be=(bv>>23)&255;ce=(cv>>23)&255
 am=tl.where(ae==0,0,(av&0x7fffff)|0x800000)
 bm=tl.where(be==0,0,(bv&0x7fffff)|0x800000)
 cm=tl.where(ce==0,0,(cv&0x7fffff)|0x800000)
 pm=am*bm;ps=tl.where((av^bv)<0,-1,1).to(tl.int64)
 cs=tl.where(cv<0,-1,1).to(tl.int64)
 pe=ae+be-254-46;cee=ce-127-23
 pt=tl.where(pm==0,-10000,ae+be-254+1)
 ct=tl.where(cm==0,-10000,ce-127)
 scale=tl.maximum(pt,ct)-60
 scale=tl.where((pm==0)&(cm==0),0,scale)
 pi,plost=_align(pm,pe,scale);ci,clost=_align(cm,cee,scale)
 total=ps*pi+cs*ci
 residual=ps*plost.to(tl.int64)+cs*clost.to(tl.int64)
 sign=tl.where(total<0,-1,1).to(tl.int64)
 sticky=residual*sign;mag=tl.abs(total)
 lead=tl.maximum(((mag.to(tl.float32).to(tl.int32,bitcast=True).to(tl.int64)>>23)&255)-127,0)
 one=tl.full((),1,tl.int64)
 lead=tl.maximum(lead-(mag<(one<<lead)).to(tl.int64),0)
 sh=lead-23;right=tl.maximum(sh,0)
 mant=tl.where(sh>=0,mag>>right,mag<<tl.maximum(-sh,0))
 low=mag&((one<<right)-1)
 half=one<<tl.maximum(right-1,0)
 up=(sh>0)&((low>half)|((low==half)&((sticky>0)|((sticky==0)&((mant&1)!=0)))))
 mant=mant+up.to(tl.int64);carry=mant>=0x1000000
 mant=tl.where(carry,mant>>1,mant)
 exponent=scale+lead+127+carry.to(tl.int64)
 bits=(exponent<<23)|(mant&0x7fffff)|tl.where(total<0,2147483648,0).to(tl.int64)
 bits=tl.where(mag==0,0,bits)
 value=bits.to(tl.int32).to(tl.float32,bitcast=True)
 tl.store(O+i,value,i<N)

def fma(a,b,c):
 a,b,c=torch.broadcast_tensors(a,b,c)
 a=a.contiguous();b=b.contiguous();c=c.contiguous()
 out=torch.empty_like(a)
 _fma_kernel[(triton.cdiv(a.numel(),256),)](a,b,c,out,a.numel(),BLOCK=256)
 return out
