"""Dense BF16 flash attention for the actual Qwen 2.1 uncached image queries."""
import torch
import triton
import triton.language as tl

@triton.jit
def dense_attention(Q,K,V,O,SQH:tl.constexpr,SQM:tl.constexpr,SKH:tl.constexpr,SKM:tl.constexpr,SVH:tl.constexpr,SVM:tl.constexpr,M:tl.constexpr,N:tl.constexpr,D:tl.constexpr,SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
 h=tl.program_id(1);rows=tl.program_id(0)*BM+tl.arange(0,BM);cols=tl.arange(0,BN);ds=tl.arange(0,D)
 q=tl.load(Q+h*SQH+rows[:,None]*SQM+ds[None,:],rows[:,None]<M,0)
 mx=tl.full((BM,),float('-inf'),tl.float32);den=tl.full((BM,),0.,tl.float32);out=tl.full((BM,D),0.,tl.float32)
 for start in range(tl.cdiv(N,BN)):
  ns=start*BN+cols
  k=tl.load(K+h*SKH+ns[None,:]*SKM+ds[:,None],ns[None,:]<N,0)
  score=tl.dot(q,k)*SCALE
  score=tl.where(ns[None,:]<N,score,float('-inf'))
  nxt=tl.maximum(mx,tl.max(score,axis=1));p=tl.exp2(score-nxt[:,None]);factor=tl.exp2(mx-nxt)
  v=tl.load(V+h*SVH+ns[:,None]*SVM+ds[None,:],ns[:,None]<N,0)
  out=out*factor[:,None]
  out=tl.dot(p.to(q.dtype),v,out)
  den=den*factor+tl.sum(p,axis=1);mx=nxt
 tl.store(O+h*M*D+rows[:,None]*D+ds[None,:],(out/den[:,None]).to(O.dtype.element_ty),rows[:,None]<M)

def attention(q,k,v,bm=64,bn=64,warps=4):
 # [B,H,N,D], batch one for the measured fixture.
 assert q.shape[0]==k.shape[0]==v.shape[0]==1
 m,d=q.shape[-2:];n=k.shape[-2];h=q.shape[1]
 out=torch.empty((1,h,m,d),dtype=q.dtype,device=q.device)
 dense_attention[(triton.cdiv(m,bm),h)](q,k,v,out,q.stride(1),q.stride(2),k.stride(1),k.stride(2),v.stride(1),v.stride(2),m,n,d,d**-.5*1.4426950408889634,bm,bn,num_warps=warps,num_stages=1)
 return out
