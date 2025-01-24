import torch as tc
from torch import nn
class BinarySource(nn.Module):
  """Layer generating random binary tensors.
  Parameters----------
  dtype : tc.DType: Defines the output datatype of the layer.
  seed : int or None: seed for prng used to gen the bits.
      Set to `None` for random init of the RNG.
  Input--
  shape : 1D array, int: desired shape of output tensor.
  Output---
  : `shape`, `dtype`: Tensor filled with random binary values.
  """
  def __init__(self,dtype=tc.float32,device='cpu'):
    super().__init__()
    self.dtype= dtype
    self.device=device
  def forward(self, inputs):
    return tc.randint(0, 2,size=inputs,device=self.device,dtype=self.dtype)