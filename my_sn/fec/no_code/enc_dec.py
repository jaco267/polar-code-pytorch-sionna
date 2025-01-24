import torch as tc
from torch import nn
class no_encoder(nn.Module):
  def __init__(self):
    super().__init__()    
  def forward(self, inputs):
    return inputs
class no_decoder(nn.Module):
  def __init__(self):
    super().__init__()    
  def forward(self, llr):
    return tc.where(llr>0,1.,0.)