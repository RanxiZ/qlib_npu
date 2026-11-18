"""Isolated CUDA-shaped validation IC reductions; not installed in Qlib.
Mean and Welford schedules follow ATen Reduce.cuh/SharedReduceOps.h.
Restricted to the observed contiguous FP32 daily lengths 178..280.
"""
import numpy as np
import torch
from .triton_ieee_fma import fma
from .triton_ieee_div import div

def _layout(starts,vec,lanes,device):
 starts=np.asarray(starts,dtype=np.int64);lo=starts[:-1];n=np.diff(starts)
 if np.any((n<178)|(n>280)):raise ValueError('Unverified daily reduction length')
 width=np.array([1<<int(v//vec).bit_length()-1 for v in n],dtype=np.int64)
 shift=lo%vec;head=np.where(shift,vec-shift,0)
 def t(a):return torch.as_tensor(a,device=device)
 return t(lo[:,None,None]),t(n[:,None,None]),t(width[:,None,None]),t(shift[:,None,None]),t(head[:,None,None]),torch.arange(lanes,device=device)[None,:,None],torch.arange(vec,device=device)[None,None,:]

def daily_mean(v,starts):
 lo,n,w,shift,head,lane,j=_layout(starts,4,64,v.device)
 a=torch.zeros((len(starts)-1,64,4),device=v.device)
 def load(idx,mask):return torch.where(mask,v[idx.clamp(0,v.numel()-1)],0.)
 idx=lo+lane-shift
 a[:,:,0]=load(idx,(shift>0)&(lane>=shift)&(lane<4))[:,:,0]
 end=n-head;full=end//4*4
 for it in range(2):
  rel=(lane+it*w)*4+j;idx=lo+head+rel
  a=a+load(idx,(lane<w)&(rel<full))
 tailidx=lo+head+full+lane
 a[:,:,0]=a[:,:,0]+load(tailidx,(lane<(end%4)))[:,:,0]
 s=((a[:,:,0]+a[:,:,1])+a[:,:,2])+a[:,:,3]
 for off in [32,16,8,4,2,1]:s=s[:,:off]+s[:,off:2*off]
 rec=torch.as_tensor((1/np.diff(starts)).astype(np.float32),device=v.device)
 return s[:,0]*rec

def daily_std(v,starts):
 lo,n,w,shift,head,lane,j=_layout(starts,2,128,v.device)
 m=torch.zeros((len(starts)-1,128,2),device=v.device);q=torch.zeros_like(m);c=torch.zeros_like(m)
 def add(m,q,c,x,valid):
  nc=c+1;delta=x-m;nm=m+div(delta,nc);nq=fma(delta,x-nm,q)
  return torch.where(valid,nm,m),torch.where(valid,nq,q),torch.where(valid,nc,c)
 idx=(lo+lane-shift).clamp(0,v.numel()-1);valid=(shift>0)&(lane==1)&(j==0)
 m,q,c=add(m,q,c,v[idx].expand_as(m),valid)
 end=n-head;full=end//2*2
 for it in range(2):
  rel=(lane+it*w)*2+j;idx=(lo+head+rel).clamp(0,v.numel()-1);valid=(lane<w)&(rel<full)
  m,q,c=add(m,q,c,v[idx],valid)
 idx=(lo+head+full).clamp(0,v.numel()-1);valid=(end%2>0)&(lane==0)&(j==0)
 m,q,c=add(m,q,c,v[idx].expand_as(m),valid)
 def merge(am,av,ac,bm,bv,bc):
  count=ac+bc;delta=bm-am;r=div(bc,torch.where(count>0,count,1.))
  nm=fma(delta,r,am);nq=fma((delta*delta)*ac,r,av+bv)
  return torch.where(ac==0,bm,torch.where(bc==0,am,nm)),torch.where(ac==0,bv,torch.where(bc==0,av,nq)),count
 m,q,c=merge(m[:,:,0],q[:,:,0],c[:,:,0],m[:,:,1],q[:,:,1],c[:,:,1])
 for off in [64,32,16,8,4,2,1]:m,q,c=merge(m[:,:off],q[:,:off],c[:,:off],m[:,off:2*off],q[:,off:2*off],c[:,off:2*off])
 denom=torch.as_tensor(np.diff(starts).astype(np.float32)-1,device=v.device)
 return div(q[:,0],denom).sqrt()

def normalize(v,starts,means,stds):
 n=np.diff(starts);ids=torch.as_tensor(np.repeat(np.arange(len(n)),n),device=v.device)
 rec=torch.as_tensor((np.float32(1)/np.sqrt(n).astype(np.float32)),device=v.device)
 return div((v-means[ids])*rec[ids],stds[ids])

def daily_dot(p,y,starts):
 n=np.diff(starts);lo=torch.as_tensor(np.asarray(starts[:-1]),device=p.device)[:,None]
 ids=lo+torch.arange(512,device=p.device)[None,:];valid=ids<torch.as_tensor(np.asarray(starts[1:]),device=p.device)[:,None]
 idx=ids.clamp(0,p.numel()-1)
 products=torch.where(valid,p[idx]*y[idx],0.).reshape(len(n),2,128,2)
 s=products[:,:,:,0]+products[:,:,:,1]
 for off in [64,32,16,8,4,2,1]:s=s[:,:,:off]+s[:,:,off:2*off]
 return s[:,0,0]+s[:,1,0]

def validation_ic(pred,label,starts,return_stages=False):
 pm=daily_mean(pred,starts);ym=daily_mean(label,starts)
 ps=daily_std(pred,starts);ys=daily_std(label,starts)
 pn=normalize(pred,starts,pm,ps);yn=normalize(label,starts,ym,ys)
 dots=daily_dot(pn,yn,starts)
 valid=(ps>=1e-12)&(ys>=1e-12);total=torch.zeros((),device=pred.device)
 for j in range(len(starts)-1):total=total+torch.where(valid[j],dots[j],0.)
 count=int(valid.sum().item())
 if count==0:raise ValueError('No eligible validation days')
 metric=total*float(np.float32(1.0/count))
 return (metric,dict(pred_mean=pm,y_mean=ym,pred_std=ps,y_std=ys,dot=dots,pn=pn,yn=yn)) if return_stages else metric
