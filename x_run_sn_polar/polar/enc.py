import numpy as np
import torch as tc
from torch import nn
from my_sn.fec.polar.utils import generate_5g_ranking
from my_sn.fec.crc import CRCEncoder
#* done   

class PolarEncoder(nn.Module):
  #`k`info bits & `frozen set` (idxes of frozen pos) specified by `frozen_pos`
  def __init__(self,frozen_pos, #np int:`n-k` frozen indices,
    # ex. n=8, k=4, frozen_pos=[0,1,2,4] 
    n, #128:int: codeword len. 
    G,
    dtype=tc.float32,
    device='cpu'
  ):
    super().__init__()
    self.device= device
    self.dtype= dtype
    assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."
    self._k = n - len(frozen_pos) #128-53=75#Number of info bits.
    self._n = n  #128#Codeword len.
    self._frozen_pos=frozen_pos #[int,]#Frozen positions for Polar decoding
    self.info_pos = np.setdiff1d(np.arange(self._n),frozen_pos)#len=64,(not frozon) info positions for Polar encoding
    # ex. n=8, k=4, frozen_pos=[3,5,6,7]
    assert self._k==len(self.info_pos), "invalid info_pos generated."
    self.G_ = G
  @property
  def k(self):return self._k #"""Number of information bits."""
  def forward(self, u):#inputs(tc.fp32):[bs,k] info bits to be encoded.
    """returns polar encoded codewords for given info bits inputs."""
    bs = u.shape[0];  assert u.shape[-1]==self._k,"Last dim must be len k."
    c = tc.zeros([bs,self._n], dtype=self.dtype,device=self.device) #codeword, ex.(bs=100,n=8)
    info_pos_tc = tc.from_numpy(self.info_pos) #ex.n=8,k=4->info_pos=[3,5,6,7] 
    c[...,info_pos_tc] = u#copy info bits to info pos; other pos are frozen (=0)
    '''ex. for N=8,k=4  info_pos = [3,5,6,7]
    u = [u0,u1,u2,u3]  
    info_pos=[0,0,0, 1,0, 1, 1, 1]
    c =      [0,0,0,u0,0,u1,u2,u3]
    '''
    # out = self.G_matrix(bs,c,device=self.device)  #c, out (bs,n)
    out = (c@self.G_%2).to(dtype=self.dtype)
    return out
  