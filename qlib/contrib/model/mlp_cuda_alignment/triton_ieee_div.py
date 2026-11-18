"""Experimental fused RN-even normal FP32 division; zeros supported."""
import torch
import triton
import triton.language as tl

@triton.jit
def _div_kernel(A,B,O,N,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 av=tl.load(A+i,i<N,other=0).to(tl.int32,bitcast=True).to(tl.int64)
 bv=tl.load(B+i,i<N,other=1).to(tl.int32,bitcast=True).to(tl.int64)
 ae=(av>>23)&255;be=(bv>>23)&255
 am=(av&0x7fffff)|0x800000;bm=(bv&0x7fffff)|0x800000
 lower=am<bm;sh=23+lower.to(tl.int64)
 rem=am<<sh;q=tl.full((BLOCK,),0,tl.int64)
 for bit in tl.static_range(24,-1,-1):
  trial=bm<<bit;take=rem>=trial
  rem=tl.where(take,rem-trial,rem);q=q|(take.to(tl.int64)<<bit)
 q=q+((rem*2>bm)|((rem*2==bm)&((q&1)!=0))).to(tl.int64)
 carry=q>=0x1000000;q=tl.where(carry,q>>1,q)
 exp=ae-be+127-lower.to(tl.int64)+carry.to(tl.int64)
 sign=(av^bv)&2147483648
 bits=sign|(exp<<23)|(q&0x7fffff);bits=tl.where(ae==0,sign,bits)
 tl.store(O+i,bits.to(tl.int32).to(tl.float32,bitcast=True),i<N)

def div(a,b):
 a,b=torch.broadcast_tensors(a,b);a=a.contiguous();b=b.contiguous()
 out=torch.empty_like(a)
 _div_kernel[(triton.cdiv(a.numel(),256),)](a,b,out,a.numel(),BLOCK=256)
 return out
