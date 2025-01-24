import numpy as np
import torch as tc
from torch import nn
import math
class SC_Dec(nn.Module):
  """SC decoder [Arikan_Polar]_ 
  implementation follows `recursive tree` [Gross_Fast_SCL]_ terminology and 
  combines nodes for increased throughputs without changing outcome of algo.
  """
  def __init__(self, frozen_pos, n, 
    output_dtype=tc.float32,device='cpu',mode='llr'):
    super().__init__()
    self.output_dtype = output_dtype# tc.fp32.
    # store internal attributes
    self.n = n #codeword len.  #  n=k*r
    self.frozen_pos = frozen_pos #`n-k` indices of Frozen positions for Polar decoding
    self.k = self.n - len(self.frozen_pos)# info bits num.
    #Info bit pos for Polar encoding
    self.info_pos = np.setdiff1d(np.arange(self.n), self.frozen_pos)
    assert self.k==len(self.info_pos), "Internal error: invalid " "info_pos generated."
    self.llr_max = 30. #Maximum LLR value for internal calculations. (uncritical for SC dec)
    #create a frozen bit vector for simpler encoding
    self._frozen_ind = np.zeros(self.n)
    self._frozen_ind[self.frozen_pos] = 1
    self._use_fast_sc = False  # enable graph pruning
    self.complexity = None  
    self.find_complexity = False
    self.mode = mode
    self._cw_ind = np.arange(self.n)
    self.kern_size = 2  
    self._n_stages = int(math.log(n,self.kern_size))
    self.print_opt = False
  def _cn_op_tf(self, x, y):
    """f_function  See [Stimming_LLR]_ and [Hashemi_SSCL]_ for detailed equations."""
    x_in = tc.clip(x,min=-self.llr_max,max=self.llr_max)
    y_in = tc.clip(y,min=-self.llr_max,max=self.llr_max)
    if self.mode == "llr":
      # avoid division for numerical stability
      llr_out = tc.log(1 + tc.exp(x_in + y_in))
      llr_out -= tc.log(tc.exp(x_in) + tc.exp(y_in))
      # f function ln((l1l2+1)/(l1+l2)) = ln(e^(L1+L2)+1) - ln(e^L1 + e^L2) 
    elif self.mode == "max":
      llr_out = tc.sign(x_in)*tc.sign(y_in)*tc.min(tc.abs(x_in),tc.abs(y_in))
    else: 
      raise Exception('error...') 
    llr_out = tc.sign(x_in)*tc.sign(y_in)*tc.min(tc.abs(x_in),tc.abs(y_in))
    # breakpoint()
    return llr_out
  def _vn_op_tf(self, x, y, u_hat):
    """VN update for LLR inputs. g function"""
    #ln(g(l1,l2)) = ln(g(e^L1,e^L2)) =  L2+L1 if u^ = 0  = L2-L1 if u^ = 1  = L2 + (1-2u)L1
    out =  (1-2*u_hat)*x + y
    return out
  def _polar_decode_sc_tf(self,cw_ind):
    """Rec SC dec fn. https://youtu.be/YD4fF12TVB4?t=1429"""
    n = len(cw_ind)
    stage_ind = int(math.log(n,self.kern_size))
    if n>1:# branch if leaf is not reached yet
      cw_ind_left = cw_ind[0:int(n/2)] #*(n//2=32,)[0~31]
      cw_ind_right = cw_ind[int(n/2):] #*(n//2=32,)[32~63]
      #*-----split llr-----
      llr_left = self.msg_llr[:, stage_ind, cw_ind_left]
      llr_right = self.msg_llr[:,stage_ind, cw_ind_right]
      #*-----decode phase i(0~1) + call recurr------
      # upper path  #* kernel processing phase 1  
      c_val = self._cn_op_tf(llr_left,llr_right)
      self.msg_llr[:, stage_ind-1, cw_ind_left] = c_val
      self._polar_decode_sc_tf(cw_ind_left) #* input f1,f2  frozen upper
      # u_hat_left_up = self.msg_uhat[]
      u_hat_left_up = self.msg_uhat[:, stage_ind-1, cw_ind_left]
      assert  tc.all(llr_left==self.msg_llr[:, stage_ind, cw_ind_left])
      assert  tc.all(llr_right==self.msg_llr[:, stage_ind, cw_ind_right])
      
      v_val = self._vn_op_tf(
        llr_left,llr_right,u_hat_left_up)
      self.msg_llr[:, stage_ind-1, cw_ind_right]=v_val
      if v_val.shape[1] == 1 and self.print_opt:#and print_op
        print(c_val[0].item(),end=' ')
        print(v_val[0].item(),end=' ')
      # lower path
      self._polar_decode_sc_tf(cw_ind_right)
      #*--------reencode-------- combine u_hat from both branches     
      u_hat_left_up = self.msg_uhat[:, stage_ind-1, cw_ind_left]
      u_hat_right_up = self.msg_uhat[:, stage_ind-1, cw_ind_right]
      # u_hat_left_up XOR u_hat_right_up
      u_hat_left =  (u_hat_left_up != u_hat_right_up) + 0
      u_hat0 = tc.concatenate([u_hat_left, u_hat_right_up], dim=-1)
      # provide u_hat for next higher stage
      self.msg_uhat[:, stage_ind,  cw_ind] = u_hat0
    else: # if leaf is reached perform basic decoding op (=decision)
      if self._frozen_ind[cw_ind]==1: # position is frozen, so we know u_hat is 0
          self.msg_uhat[:,0,cw_ind] = 0
      else: # otherwise hard decide  #posidtion not frozen 
          llr_ch0 = self.msg_llr[:,0,cw_ind]
          u_hat = 0.5 * (1. - tc.sign(llr_ch0))  
          #remove "exact 0 llrs" leading to u_hat=0.5
          u_hat = tc.where(u_hat==0.5,tc.ones_like(u_hat),u_hat)
          self.msg_uhat[:,0,cw_ind] = u_hat
  def _decode_batch(self,llr_ch):
    '''ex. n = 8  3 stage
    s0 s1 s2 s3        s0 s1 s2 s3
    msg_llr            msg_uhat
    0        llr[0]    0
    ...                ...
    0        llr[7]    0  
    '''
    bs = llr_ch.shape[0]
    self.msg_uhat = tc.zeros([bs,self._n_stages+1,self.n]) #*hard val
    self.msg_llr = tc.zeros([bs,self._n_stages+1,self.n]) #* soft val
    self.msg_llr[:,self._n_stages,:] = llr_ch
    self._polar_decode_sc_tf(self._cw_ind)
    return self.msg_uhat[:,0,:]
  def forward(self, inputs):#(tc.fp32): shape `[...,n]`2+D tensor with channel LLR values (as logits).
    # return tc.fp32:shape `[...,k]`2+D tensor with hard-decided estimations of all `k` info bits.
    """Performs sc dec and returns estimated info bits."""
    self.complexity = 0
    inputs = inputs.to(dtype=tc.float32)
    assert inputs.shape[-1]== self.n,"Last input dim must be of len n."
    assert len(inputs.shape)> 1
    input_shape = inputs.shape
    llr_ch = inputs.reshape([-1, self.n])
    llr_ch = -1. * llr_ch # logits are converted into "true" llrs
    # breakpoint()
    u_hat_n = self._decode_batch(llr_ch)
    
    # recover k info bit positions
    u_hat = u_hat_n[...,self.info_pos]  
    # and reconstruct input shape
    output_shape = list(input_shape)
    output_shape[-1] = self.k
    output_shape[0] = -1 # first dim can be dynamic (None)
    u_hat_reshape = u_hat.reshape(output_shape)
    return u_hat_reshape.to(dtype=self.output_dtype)