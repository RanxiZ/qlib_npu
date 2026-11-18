"""Experimental fusion of the verified four-TF32-product accumulation rule."""
import torch
import triton
import triton.language as tl

@triton.jit
def _mma4_kernel(A,B,C,O,N,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 r=tl.arange(0,4)
 a=tl.load(A+r[:,None]*N+i[None,:],i[None,:]<N,other=0)
 b=tl.load(B+r[:,None]*N+i[None,:],i[None,:]<N,other=0)
 c=tl.load(C+i,i<N,other=0)
 products=a*b
 bits=products.to(tl.int32,bitcast=True).to(tl.int64)
 cb=c.to(tl.int32,bitcast=True).to(tl.int64)
 exp=(bits>>23)&255;ce=(cb>>23)&255
 mant=tl.where(exp==0,0,(bits&0x7fffff)|0x800000)
 cm=tl.where(ce==0,0,(cb&0x7fffff)|0x800000)
 ab=a.to(tl.int32,bitcast=True).to(tl.int64)
 bb=b.to(tl.int32,bitcast=True).to(tl.int64)
 ae=(ab>>23)&255;be=(bb>>23)&255
 raw=tl.where((ae==0)|(be==0),0,ae+be-127)
 top=tl.maximum(tl.max(raw,axis=0),ce)
 shift=top[None,:]-exp;cs=top-ce
 right=tl.minimum(tl.maximum(shift,0),62)
 left=tl.minimum(tl.maximum(-shift,0),62)
 aligned=tl.where(shift>=0,(mant<<1)>>right,(mant<<1)<<left)
 cr=tl.minimum(tl.maximum(cs,0),62);cl=tl.minimum(tl.maximum(-cs,0),62)
 ca=tl.where(cs>=0,(cm<<1)>>cr,(cm<<1)<<cl)
 total=tl.sum(tl.where(bits<0,-aligned,aligned),axis=0)+tl.where(cb<0,-ca,ca)
 mag=tl.abs(total)
 lead=tl.maximum(((mag.to(tl.float32).to(tl.int32,bitcast=True).to(tl.int64)>>23)&255)-127,0)
 one=tl.full((),1,tl.int64)
 lead=tl.maximum(lead-(mag<(one<<lead)).to(tl.int64),0)
 sh=lead-23
 norm=tl.where(sh>=0,mag>>tl.maximum(sh,0),mag<<tl.maximum(-sh,0))
 out_exp=top-24+lead
 out=(out_exp<<23)|(norm&0x7fffff)|tl.where(total<0,2147483648,0).to(tl.int64)
 out=tl.where(mag==0,0,out)
 tl.store(O+i,out.to(tl.int32).to(tl.float32,bitcast=True),i<N)

def mma4(a,b,c):
 if a.shape!=b.shape or a.shape[0]!=4 or a.shape[1]!=c.numel():
  raise ValueError('Expected a,b=(4,N), c=(N,)')
 a=a.contiguous();b=b.contiguous();c=c.contiguous()
 out=torch.empty_like(c)
 _mma4_kernel[(triton.cdiv(c.numel(),128),)](a,b,c,out,c.numel(),BLOCK=128)
 return out

@triton.jit
def _mma4_values(a,b,c):
 products=a*b
 bits=products.to(tl.int32,bitcast=True).to(tl.int64)
 cb=c.to(tl.int32,bitcast=True).to(tl.int64)
 exp=(bits>>23)&255;ce=(cb>>23)&255
 mant=tl.where(exp==0,0,(bits&0x7fffff)|0x800000)
 cm=tl.where(ce==0,0,(cb&0x7fffff)|0x800000)
 ab=a.to(tl.int32,bitcast=True).to(tl.int64)
 bb=b.to(tl.int32,bitcast=True).to(tl.int64)
 ae=(ab>>23)&255;be=(bb>>23)&255
 raw=tl.where((ae==0)|(be==0),0,ae+be-127)
 top=tl.maximum(tl.max(raw,axis=0),ce)
 shift=top[None,:]-exp;cs=top-ce
 right=tl.minimum(tl.maximum(shift,0),62)
 left=tl.minimum(tl.maximum(-shift,0),62)
 aligned=tl.where(shift>=0,(mant<<1)>>right,(mant<<1)<<left)
 cr=tl.minimum(tl.maximum(cs,0),62);cl=tl.minimum(tl.maximum(-cs,0),62)
 ca=tl.where(cs>=0,(cm<<1)>>cr,(cm<<1)<<cl)
 total=tl.sum(tl.where(bits<0,-aligned,aligned),axis=0)+tl.where(cb<0,-ca,ca)
 mag=tl.abs(total)
 lead=tl.maximum(((mag.to(tl.float32).to(tl.int32,bitcast=True).to(tl.int64)>>23)&255)-127,0)
 one=tl.full((),1,tl.int64)
 lead=tl.maximum(lead-(mag<(one<<lead)).to(tl.int64),0)
 sh=lead-23
 norm=tl.where(sh>=0,mag>>tl.maximum(sh,0),mag<<tl.maximum(-sh,0))
 out_exp=top-24+lead
 out=(out_exp<<23)|(norm&0x7fffff)|tl.where(total<0,2147483648,0).to(tl.int64)
 out=tl.where(mag==0,0,out)
 return out.to(tl.int32).to(tl.float32,bitcast=True)
