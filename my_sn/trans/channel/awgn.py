"""Layer for simulating an AWGN channel"""
import torch as tc
from torch import nn
from my_sn.utils import expand_to_rank,complex_normal
class AWGN(nn.Module):
  r"""adds complex AWGN noise with variance ``no`` to input.
  The noise has variance `no/2`.
  Input-----(x, no) :Tuple:
   x : Tensor, tc.complex :Channel input
   no : Scalar,tc.fp, noise power `no`
  Output-------
   y : same shape as `x`,tc.complex:Channel output
  """
  def __init__(self,device = 'cpu'):
    super().__init__()
    #dtype:Complex: datatype for internal calcu & output dtype.
    self._real_dtype = tc.float32
    self.device = device
  def forward(self, inputs):
    x, no = inputs  #x:tc.complex64
    #gen tensors of real-val Gauss noise for each complex dim.
    noise = complex_normal(x.shape,device=self.device)
    # Add extra dims for broadcasting
    no = expand_to_rank(no,target_rank=len(x.shape),axis=-1)
    # todo  maybe it's possible to delete expand_rand
    # Apply variance scaling
    noise *= tc.sqrt(no.to(dtype=self._real_dtype)).to(dtype=noise.dtype,device=self.device)
    y = x + noise   # Add noise to input
    return y
