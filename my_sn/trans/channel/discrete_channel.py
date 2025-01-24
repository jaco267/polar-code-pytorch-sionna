import torch as tc
from torch import nn
from my_sn.utils import expand_to_rank,complex_normal

class BinaryMemorylessChannel(nn.Module):
  def __init__(self, return_llrs=False, bipolar_input=False, 
    llr_max=100.,dtype = tc.float32,device = 'cpu'):
    super().__init__()
    self.dtype = dtype
    assert isinstance(return_llrs, bool), "return_llrs must be bool."
    self._return_llrs = return_llrs
    assert isinstance(bipolar_input, bool), "bipolar_input must be bool."
    self._bipolar_input = bipolar_input
    assert llr_max>=0., "llr_max must be a positive scalar value."
    self._llr_max = tc.tensor(llr_max).to(dtype=self.dtype)
    if self._return_llrs:assert dtype in (tc.float16, tc.float32, tc.float64),"LLR outputs require non-integer dtypes."
    else:
        if self._bipolar_input:
            assert dtype in (tc.float16, tc.float32, tc.float64,tc.int8, tc.int16, tc.int32, tc.int64),"Only, signed dtypes are supported for bipolar inputs."
        else:
            assert dtype in (tc.float16, tc.float32, tc.float64,tc.uint8, tc.uint16, tc.uint32, tc.uint64,
                tc.int8, tc.int16, tc.int32, tc.int64),"Only, real-valued dtypes are supported."
    self._check_input = True # check input for consistency (i.e., binary)
    self._eps = 1e-9 # small additional term for numerical stability
    self._temperature = 0.1 # for Gumble-softmax
  @property
  def llr_max(self):return self._llr_max#"""Maximum value used for LLR calculations."""
  @llr_max.setter
  def llr_max(self, value):##"""Maximum value used for LLR calculations."""
      assert value>=0, 'llr_max cannot be negative.'
      self._llr_max = value.to(dtype=tc.float32) 
  @property
  def temperature(self):return self._temperature #"""Temperature for Gumble-softmax trick."""
  @temperature.setter
  def temperature(self, value):#"""Temperature for Gumble-softmax trick."""
      assert value>=0, 'temperature cannot be negative.'
      self._temperature = value.to(dtype=tc.float32)
  def _check_inputs(self, x):
    """Check input x for consistency, i.e., verify
    that all values are binary of bipolar values."""
    x = x.to(tc.float32)
    if self._check_input:
      if self._bipolar_input: # allow -1 and 1 for bipolar inputs
          values = (tc.tensor(-1).to(x.dtype),tc.tensor(1).to(x.dtype))
      else: # allow 0,1 for binary input
          values = (tc.tensor(0).to(x.dtype),tc.tensor(1).to(x.dtype))
      assert tc.all(tc.bitwise_or(x== values[0], x== values[1])==1),"Input must be binary."
      # input datatype consistency should be only evaluated once
      self._check_input = False
  def _ste_binarizer(self, x):
    """Straight through binarizer to quantize bits to int values."""
    #def grad(upstream):return upstream #"""identity in backward direction"""
    # hard-decide in forward path
    z = tc.where(x<.5, 0., 1.)
    return z#, grad
  def _sample_errors(self, pb, shape):
    """Samples binary error vector with given error probability e.
    This function is based on the Gumble-softmax "trick" to keep the
    sampling differentiable."""
    # this implementation follows https://arxiv.org/pdf/1611.01144v5.pdf
    # and https://arxiv.org/pdf/1906.07748.pdf
    u1 = tc.rand(shape).to(tc.float32)
    u2 = tc.rand(shape).to(tc.float32)
    u = tc.stack((u1, u2), dim=-1)
    # sample Gumble distribution
    q = - tc.log(- tc.log(u + self._eps) + self._eps)
    p = tc.stack((pb,1-pb), dim=-1)
    p = expand_to_rank(p, len(q.shape), axis=0)
    p = tc.broadcast_to(p, q.shape)
    a = (tc.log(p + self._eps) + q) / self._temperature
    # apply softmax
    
    e_cat = tc.softmax(a,dim=-1)
    # binarize final values via straight-through estimator
    return self._ste_binarizer(e_cat[...,0]) # only take first class
  
  def forward(self):
    breakpoint()
class BinaryErasureChannel(BinaryMemorylessChannel):
  def __init__(self, return_llrs=False, bipolar_input=False, llr_max=100.,dtype=tc.float32):
    super().__init__(return_llrs=return_llrs,bipolar_input=bipolar_input,
        llr_max=llr_max,dtype=dtype)
    # also exclude uints, as -1 indicator for erasures does not exist
    assert dtype in (tc.float16, tc.float32, tc.float64,tc.int8, tc.int16, tc.int32, tc.int64),\
            "Unsigned integers are currently not supported."
  def forward(self,inputs):
    """Apply erasure channel to inputs."""
    x, pb = inputs
    # clip for numerical stability
    pb = pb.to(tc.float32) # Gumble requires float dtypes
    pb = tc.clip(pb, 0., 1.)
    self._check_inputs(x)# check x for consistency (binary, bipolar)
    e = self._sample_errors(pb, x.shape)# sample erasure pattern
    # if LLRs should be returned
    # remark: the Sionna logit definition is llr = log[p(x=1)/p(x=0)]
    if self._return_llrs:
        if not self._bipolar_input:
            x = 2 * x -1
        x *= self._llr_max.to(x.dtype) # calculate llrs
        # erase positions by setting llrs to 0
        y = tc.where(e==1, 0, x)
    else: # ternary outputs
        # the erasure indicator depends on the operation mode
        if self._bipolar_input:erased_element = 0
        else:erased_element = -1
        y = tc.where(e==0, x, erased_element)
    return y