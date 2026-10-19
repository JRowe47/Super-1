"""Complete native-order orthonormal Haar transform with exact linear adjoints."""
import numpy as np
import torch
def haar_np(x):
    a=np.asarray(x,np.float64).ravel();details=[]
    while len(a)%2==0:
        details.append((a[::2]-a[1::2])/np.sqrt(2))
        a=(a[::2]+a[1::2])/np.sqrt(2)
    return np.concatenate([a]+details[::-1])
def inverse_np(c):
    c=np.asarray(c,np.float64).ravel();n=len(c);small=n
    while small%2==0:small//=2
    a=c[:small].copy();off=small
    while off<n:
        d=c[off:off+len(a)];off+=len(a);b=np.empty(2*len(a))
        b[::2]=(a+d)/np.sqrt(2);b[1::2]=(a-d)/np.sqrt(2);a=b
    return a
def haar(x):return torch.from_numpy(haar_np(x.detach().cpu().numpy()))
def inverse(x):return torch.from_numpy(inverse_np(x.detach().cpu().numpy()))
class Decode(torch.autograd.Function):
    @staticmethod
    def forward(ctx,c):return inverse(c)
    @staticmethod
    def backward(ctx,gradient):return haar(gradient)
def decode(c):return Decode.apply(c)
