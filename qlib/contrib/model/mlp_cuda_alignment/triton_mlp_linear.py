"""Experimental shape-specific fusion of validated CUDA MLP matrix schedules."""
import torch
import triton
import triton.language as tl
from .triton_mma4 import _mma4_values

@triton.jit
def _rna(x):
 return ((x.to(tl.int32,bitcast=True)+4096)&(-8192)).to(tl.float32,bitcast=True)

@triton.jit
def _linear157_kernel(X,W,B,O,ROWS,TILE:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 row=i//256;col=i%256;lane=tl.arange(0,4)
 residue=157%TILE
 acc=tl.full((BLOCK,),0,tl.float32)
 for group in range(40):
  packed=group*4+lane
  k=tl.where(packed<residue,packed,packed-(TILE-residue))
  valid=(packed<residue)|((packed>=TILE)&(k<157))
  a=tl.load(X+row[None,:]*157+k[:,None],valid[:,None]&(row[None,:]<ROWS),other=0)
  b=tl.load(W+col[None,:]*157+k[:,None],valid[:,None]&(row[None,:]<ROWS),other=0)
  acc=_mma4_values(_rna(a),_rna(b),acc)
 bias=tl.load(B+col,row<ROWS,other=0)
 tl.store(O+i,acc+bias,row<ROWS)

@triton.jit
def _first_weight_kernel(G,X,P,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 split=i//40192;cell=i%40192;row=cell//157;col=cell%157
 length=tl.where(split<35,228,212);residue=length%32
 lane=tl.arange(0,4)
 acc=tl.full((BLOCK,),0,tl.float32)
 for group in range(64):
  packed=group*4+lane[:,None]
  k=tl.where(packed<residue[None,:],packed,packed-(32-residue[None,:]))
  valid=(packed<residue[None,:])|((packed>=32)&(k<length[None,:]))
  global_k=tl.minimum(tl.maximum(split[None,:]*228+k,0),8191)
  a=tl.load(G+global_k*256+row[None,:],valid&(split[None,:]<36),other=0)
  b=tl.load(X+global_k*157+col[None,:],valid&(split[None,:]<36),other=0)
  acc=_mma4_values(_rna(a),_rna(b),acc)
 tl.store(P+i,acc,split<36)

@triton.jit
def _sum_splits(P,O,N,SPLITS:tl.constexpr,BLOCK:tl.constexpr):
 i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
 total=tl.full((BLOCK,),0,tl.float32)
 for split in range(SPLITS):total=total+tl.load(P+split*N+i,i<N,other=0)
 tl.store(O+i,total,i<N)

def linear_batch(x,w,bias,chunk_rows=2048,tile_k=32):
 if x.shape[1]!=157 or tuple(w.shape)!=(256,157) or tile_k not in [16,32]:
  raise ValueError('Unverified matrix shape')
 x=x.contiguous();w=w.contiguous();bias=bias.contiguous()
 out=torch.empty((len(x),256),device=x.device,dtype=torch.float32)
 _linear157_kernel[(triton.cdiv(out.numel(),128),)](x,w,bias,out,len(x),TILE=tile_k,BLOCK=128)
 return out

def first_weight_splitk_228(g,x):
 if tuple(g.shape)!=(8192,256) or tuple(x.shape)!=(8192,157):raise ValueError('Unverified gradient shape')
 parts=torch.empty((36,256,157),device=g.device,dtype=torch.float32)
 out=torch.empty((256,157),device=g.device,dtype=torch.float32)
 _first_weight_kernel[(triton.cdiv(parts.numel(),128),)](g.contiguous(),x.contiguous(),parts,BLOCK=128)
 _sum_splits[(triton.cdiv(out.numel(),256),)](parts,out,out.numel(),SPLITS=36,BLOCK=256)
 return out
