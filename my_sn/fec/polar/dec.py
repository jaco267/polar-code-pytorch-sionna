"""Layers for Polar decoding such as SC, SCL and iterative BP decoding."""
# https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCDecoder
#https://nvlabs.github.io/sionna/api/fec.polar.html#sionna.fec.polar.decoding.PolarSCLDecoder
#[Gross_Fast_SCL] https://arxiv.org/pdf/1703.08208.pdf
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from my_sn.fec.polar import Polar5GEncoder
import numpy as np
import torch as tc
from torch import nn
from my_sn.fec.crc import CRCEncoder,CRCDecoder #todo
class SC_Dec(nn.Module):
  """SC decoder [Arikan_Polar]_ 
  implementation follows `recursive tree` [Gross_Fast_SCL]_ terminology and 
  combines nodes for increased throughputs without changing outcome of algo.
  """
  def __init__(self, frozen_pos, n, output_dtype=tc.float32,device='cpu'):
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
  def _cn_op_tf(self, x, y):
    """f_function
    Check-node update (boxplus) for LLR inputs.
    Operations are performed element-wise.
    See [Stimming_LLR]_ and [Hashemi_SSCL]_ for detailed equations.
    """
    x_in = tc.clip(x,min=-self.llr_max,max=self.llr_max)
    y_in = tc.clip(y,min=-self.llr_max,max=self.llr_max)
    # avoid division for numerical stability
    llr_out = tc.log(1 + tc.exp(x_in + y_in))
    llr_out -= tc.log(tc.exp(x_in) + tc.exp(y_in))
    # f function ln((l1l2+1)/(l1+l2)) = 
    # ln(e^(L1+L2)+1) - ln(e^L1 + e^L2) 
    return llr_out
  def _vn_op_tf(self, x, y, u_hat):
    """VN update for LLR inputs. g function"""
    #ln(g(l1,l2)) = ln(g(e^L1,e^L2)) =  L2+L1 if u^ = 0   
                                    # = L2-L1 if u^ = 1  = L2 + (1-2u)L1
    return (1-2*u_hat)*x + y
  def _polar_decode_sc_tf(self, llr_ch, frozen_ind):
    """Recursive SC dec function.
    Recursively branch dec tree & split into dec of `upper` & `lower` path until reaching a leaf node.
    returns u_hat decisions at stage `0` and bit decisions of intermediate stage `s` (i.e., re-encoded version of
    `u_hat` until current stage `s`).
    https://youtu.be/YD4fF12TVB4?t=1429   
    
    to avoid numerical issues   
    L1 = ln(l1) ,  L2 = ln(l2)  #lr (l1) , llr (L1)
    ln(f(l1,l2))---+-- L1  <-- l1
                   |
    ln(g(l1,l2))------ L2  <-- l2  

        
    ln(f(l1,l2)) = ln((l1l2+1)/(l1+l2))  
                 = ln(e^(L1+L2)+1) - ln(e^L1 + e^L2)
     f'1   f1
    -+----+----L1
     |    | f2
    ------|-+--L2  f1 and f2 in first recursive is computed in parallel
    -+------|--L3  now f1 and f2 can feed into f'1      
     |      |            
    -----------L4     

    ln(g(l1,l2)) = ln(g(e^L1,e^L2)) = L2+L1 if u^ = 0   
                                    = L2-L1 if u^ = 1
    """
    n = len(frozen_ind)# calculate current codeword length
    # branch if leaf is not reached yet
    if n>1:
      llr_ch1 = llr_ch[...,0:int(n/2)]
      llr_ch2 = llr_ch[...,int(n/2):]
      frozen_ind1 = frozen_ind[0:int(n/2)]
      frozen_ind2 = frozen_ind[int(n/2):]
      # upper path  #* kernel processing phase 1  
      x_llr1_in = self._cn_op_tf(llr_ch1, llr_ch2) #* f function computed in parallel  input L1~L4 output f1,f2
      # and call dec func (with upper half)
      u_hat1, u_hat1_up = self._polar_decode_sc_tf(x_llr1_in, frozen_ind1) #* input f1,f2  frozen upper
      #* u_hat1, same_shape as x_llr1_in
      # lower path
      x_llr2_in = self._vn_op_tf(llr_ch1, llr_ch2, u_hat1_up)  #* g functoin
      
      # and call the decoding function again (with lower half)
      u_hat2, u_hat2_up = self._polar_decode_sc_tf(x_llr2_in, frozen_ind2)
      # combine u_hat from both branches
      u_hat = tc.concat([u_hat1, u_hat2], -1) #todo
      #* u_hat = u1'u2'u3'u4'
      # calculate re-encoded version of u_hat at current stage
      # u_hat1_up = tc.math.mod(u_hat1_up + u_hat2_up, 2)
      # combine u_hat via bitwise_xor (more efficient than mod2)
      u_hat1_up_int = u_hat1_up.to(dtype=tc.int8)  #* u1u2
      u_hat2_up_int = u_hat2_up.to(dtype=tc.int8)  #* u3u4
      u_hat1_up_int = tc.bitwise_xor(u_hat1_up_int,u_hat2_up_int) #* u1+u3, u2+u4
      u_hat1_up = u_hat1_up_int.to(dtype=tc.float32)  #*
      '''                       uhat1 = u1',u2'
      u_1    ---+--- u1'---+---
                |          |    and then use uhat1 we for g function we can solve u3 u4 in parallel  (u1' solve u3'g)  (u2' solve u4' g)
      u_2    ------- u2'---|-+--    x_llr2_in = u3',u4' and then we can solve uhat2 = u3',u4' 
      u_3    ---+--- u3'-----|----  
                |            |
      u_4    ------- u4'---------
      '''
      u_hat_up = tc.concat([u_hat1_up, u_hat2_up], -1) #todo
    else: # if leaf is reached perform basic decoding op (=decision)
      if frozen_ind==1: # position is frozen, so we know u_hat is 0
          u_hat = tc.zeros_like(llr_ch[:,0]).unsqueeze(dim=-1) #todo
          u_hat_up = u_hat
      else: # otherwise hard decide  #posidtion not frozen 
          u_hat = 0.5 * (1. - tc.sign(llr_ch))  #todo
          #remove "exact 0 llrs" leading to u_hat=0.5
          u_hat = tc.where(u_hat==0.5,tc.ones_like(u_hat),u_hat)
          u_hat_up = u_hat
    return u_hat, u_hat_up
  ########### Keras layer functions###########
  def build(self, input_shape):
    """Check if shape of input is invalid."""
    assert (input_shape[-1]==self.n), "Invalid input shape."
    assert (len(input_shape)>=2), 'Inputs must have at least 2 dimensions.'
  def forward(self, inputs):#(tc.fp32): shape `[...,n]`2+D tensor with channel LLR values (as logits).
    # return tc.fp32:shape `[...,k]`2+D tensor with hard-decided estimations of all `k` info bits.
    """Successive cancellation (SC) decoding function.
    Performs sc dec and returns estimated info bits.
    Raises:ValueError: If `inputs` is not of shape `[..., n]`
           InvalidArgumentError: When rank(``inputs``)<2.
    Note:
      This func recursively unrolls SC decoding tree, thus,
      for larger values of ``n`` building decoding graph can become
      time consuming.
    """
    # internal calculations still in tc.float32
    inputs = inputs.to(dtype=tc.float32)
    assert inputs.shape[-1]== self.n,"Last input dim must be of len n."
    assert len(inputs.shape)> 1
    input_shape = inputs.shape
    llr_ch = inputs.reshape([-1, self.n])
    llr_ch = -1. * llr_ch # logits are converted into "true" llrs
    # and decode
    u_hat_n, _ = self._polar_decode_sc_tf(llr_ch, self._frozen_ind)
    # and recover the k information bit positions
    u_hat = u_hat_n[...,self.info_pos]  #todo
    # and reconstruct input shape
    output_shape = list(input_shape)
    output_shape[-1] = self.k
    output_shape[0] = -1 # first dim can be dynamic (None)
    u_hat_reshape = u_hat.reshape(output_shape)
    return u_hat_reshape.to(dtype=self.output_dtype)
class SCL_Dec(nn.Module):
  # pylint: disable=line-too-long
  """PolarSCLDecoder(frozen_pos, n, list_size=8, crc_degree=None, use_hybrid_sc=False, use_fast_scl=True, cpu_only=False, use_scatter=False, ind_iil_inv=None, return_crc_status=False, output_dtype=tf.float32, **kwargs)
  Successive cancellation list (SCL) decoder [Tal_SCL]_ for Polar codes & Polar-like codes.
  Input----- [...,n], tf.float32 tensor containing channel LLR values (as logits).
  Output------
    b_hat : [...,k], tf.float32 : hard-decided estimations of all `k` info bits.
    crc_status : [...], tf.bool: CRC status indicating if a codeword was (most likely) correctly
      recovered. only returned if ``return_crc_status`` is True.Note that false positives are possible.
  Raises:AssertErr If ``list_size`` is not a power of 2.
  Note----
    implements SCL decoder in [Tal_SCL]_ but uses LLR-based message updates
    [Stimming_LLR]_. The implementation follows notation from
    [Gross_Fast_SCL]_, [Hashemi_SSCL]_. If option `use_fast_scl` is active
    tree pruning is used and tree nodes are combined if possible (see[Hashemi_SSCL]_ for details).
    dec minimizes `control flow`,-> strong memory occupation (due to full path copy after each decision).
    A hybrid SC/SCL decoder as proposed in [Cammerer_Hybrid_SCL]_ (using SC
    instead of BP) can be activated with option ``use_hybrid_sc`` iff an
    outer CRC is available. Please note that the results are not exactly
    SCL performance caused by false pos rate of CRC.
     we assume frozen bits are set to `0`. Please note
    that - although its practical relevance is only little - setting frozen
    bits to `1` may result in `affine` codes instead of linear code as the
    `all-zero` codeword is not necessarily part of the code any more.
  """
  def __init__(self,frozen_pos, # ndarray:Array of `int` defining the ``n-k`` indices of the frozen positions.
    n,#int:Defining the codeword length.
    list_size=8,#int:Defaults to 8. Defines the list size of the decoder.
    crc_degree=None, # CRC polynomial. one of `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.
    use_hybrid_sc=False, # If True, SC dec is applied and only codewords with invalid CRC are decoded with SCL. This option requires an outer CRC specified via ``crc_degree``.
    use_fast_scl=True, #If True, Tree pruning is used to reduce dec complexity. output == non-pruned version (besides numerical differences).
    return_crc_status=False,#If True, decoder also returns CRC status indicating if a codeword was (most likely) correctly recovered.  only available if ``crc_degree`` is not None.
    output_dtype=tc.float32, #Defines output datatype of layer(internal precision remains tf.float32).
    device='cpu'
  ):
    super().__init__()
    self.device = device
    self._cpu_only = True #todo
    if output_dtype not in (tc.float16, tc.float32, tc.float64):
        raise ValueError(
            'output_dtype must be {tf.float16, tf.float32, tf.float64}.')
    self.output_dtype = output_dtype# tc.fp32.
    n = int(n) # n can be float (e.g. as result of n=k*r)
    assert len(frozen_pos)<=n, "Num. of elements in frozen_pos cannot be greater than n."
    assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."
    assert np.log2(list_size)==int(np.log2(list_size)), "list_size must be a power of 2."
    # internal decoder parameters
    self._use_fast_scl = use_fast_scl # optimize rate-0 and rep nodes
    self._use_hybrid_sc = use_hybrid_sc
    # store internal attributes
    self._n = n
    self._frozen_pos = frozen_pos
    self._k = self._n - len(self._frozen_pos)
    self._list_size = list_size
    self._info_pos = np.setdiff1d(np.arange(self._n), self._frozen_pos)
    self._llr_max = 30. # internal max LLR value (not very critical for SC)
    assert self._k==len(self._info_pos), "Internal error: invalid info_pos generated."
    # create a frozen bit vector
    self._frozen_ind = np.zeros(self._n)
    self._frozen_ind[self._frozen_pos] = 1
    self._cw_ind = np.arange(self._n)
    self._n_stages = int(np.log2(self._n)) # number of decoding stages

    # init CRC check (if needed)
    if crc_degree is not None:
        self._use_crc = True
        self._crc_decoder = CRCDecoder(CRCEncoder(crc_degree,self.k))
        self._k_crc = self._crc_decoder._encoder.crc_length
        self._crc_decoder._encoder
    else:
        self._use_crc = False
        self._k_crc = 0
    assert self._k>=self._k_crc, "Value of k is too small for given CRC_degree."
    if (crc_degree is None) and return_crc_status:
        self._return_crc_status = False
        raise ValueError("Returning CRC status requires given crc_degree.")
    else:
        self._return_crc_status = return_crc_status
    # store the inverse interleaver patter
    self._iil = False #todo we ignore _iil
    self._use_hybrid_sc = False #todo we ignore hybrid
  ####### Public methods and properties#########################################
  @property
  def n(self):return self._n #"""Codeword length."""
  @property
  def k(self):return self._k #"""Number of information bits."""
  @property
  def k_crc(self):return self._k_crc #"""Number of CRC bits."""
  @property
  def frozen_pos(self):return self._frozen_pos #"""Frozen positions for Polar decoding."""
  @property
  def info_pos(self):return self._info_pos #"""Information bit positions for Polar encoding."""
  @property
  def llr_max(self):return self._llr_max #"""Maximum LLR value for internal calculations."""
  @property
  def list_size(self):return self._list_size #"""List size for SCL decoding."""
  ##### Helper functions for the TF decoder#####################################
  def _update_rate0_code(self, msg_pm, msg_uhat, msg_llr, cw_ind):pass  
  def _update_rep_code(self, msg_pm, msg_uhat, msg_llr, cw_ind):pass
  def _update_single_bit(self, ind_u, msg_uhat):pass
  def _update_pm(self, ind_u, msg_uhat, msg_llr, msg_pm):pass
  def _sort_decoders(self, msg_pm, msg_uhat, msg_llr):pass
  def _cn_op(self, x, y):pass
  def _vn_op(self, x, y, u_hat):pass
  def _duplicate_paths(self, msg_uhat, msg_llr, msg_pm):pass
  def _update_left_branch(self, msg_llr, stage_ind, cw_ind_left,cw_ind_right):pass
  def _update_right_branch(self, msg_llr, msg_uhat, stage_ind, cw_ind_left, cw_ind_right):pass
  def _update_branch_u(self, msg_uhat, stage_ind, cw_ind_left, cw_ind_right):  pass
  def _polar_decode_scl(self, cw_ind, msg_uhat, msg_llr, msg_pm):pass
  def _decode_tf(self, llr_ch):pass
  ####### Helper functions for Numpy decoder####################################
  def _update_rate0_code_np(self, cw_ind):
    """Update rate-0 (i.e., all frozen) sub-code at pos ``cw_ind`` in Numpy.
    See Eq. (26) in [Hashemi_SSCL]_."""
    n = len(cw_ind)
    stage_ind = int(np.log2(n))
    # update PM for each batch sample
    ind = np.expand_dims(self._dec_pointer, axis=-1)
    llr_in = np.take_along_axis(self.msg_llr[:, :, stage_ind, cw_ind],
        ind,axis=1)
    llr_clip = np.maximum(np.minimum(llr_in, self._llr_max), -self._llr_max)
    pm_val = np.log(1 + np.exp(-llr_clip))
    self.msg_pm += np.sum(pm_val, axis=-1)
  def _update_rep_code_np(self, cw_ind):
      """Update rep. code (i.e., only rightmost bit is non-frozen)
      sub-code at position ``ind_u`` in Numpy.
      See Eq. (31) in [Hashemi_SSCL]_.
      """
      n = len(cw_ind)
      stage_ind = int(np.log2(n))
      bs = self._dec_pointer.shape[0]
      # update PM
      llr = np.zeros([bs, 2*self._list_size, n])
      for i in range(bs):
          llr_i = self.msg_llr[i, self._dec_pointer[i, :], stage_ind, :]
          llr[i, :, :] = llr_i[:, cw_ind]
      # upper branch has negative llr values (bit is 1)
      llr[:, self._list_size:, :] = - llr[:, self._list_size:, :]
      llr_in = np.maximum(np.minimum(llr, self._llr_max), -self._llr_max)
      pm_val = np.sum(np.log(1 + np.exp(-llr_in)), axis=-1)
      self.msg_pm += pm_val
      for i in range(bs):
          ind_dec = self._dec_pointer[i, self._list_size:]
          for j in cw_ind:
              self.msg_uhat[i, ind_dec, stage_ind, j] = 1
      # branch last bit and update pm at pos cw_ind[-1]
      self._update_single_bit_np([cw_ind[-1]])
      self._sort_decoders_np()
      self._duplicate_paths_np()
  def _update_single_bit_np(self, ind_u):
      """Update single bit at position ``ind_u`` of all decoders in Numpy."""
      if self._frozen_ind[ind_u]==0: # position is non-frozen
          ind_dec = np.expand_dims(self._dec_pointer[:, self._list_size:],
                                   axis=-1)
          uhat_slice = self.msg_uhat[:, :, 0, ind_u]
          np.put_along_axis(uhat_slice, ind_dec, 1., axis=1)
          self.msg_uhat[:, :, 0, ind_u] = uhat_slice
  def _update_pm_np(self, ind_u):
    """ Update path metric of all dec at bit pos `ind_u` in Numpy.
    We apply Eq. (10) from [Stimming_LLR]_."""
    ind = np.expand_dims(self._dec_pointer, axis=-1)
    u_hat = np.take_along_axis(self.msg_uhat[:, :, 0, ind_u], ind, axis=1)
    u_hat = np.squeeze(u_hat, axis=-1)
    llr_in = np.take_along_axis(self.msg_llr[:, :, 0, ind_u], ind, axis=1)
    llr_in = np.squeeze(llr_in, axis=-1)
    llr_clip = np.maximum(np.minimum(llr_in, self._llr_max), -self._llr_max)
    self.msg_pm += np.log(1 + np.exp(-np.multiply((1-2*u_hat), llr_clip)))
  def _sort_decoders_np(self):
      """Sort decoders according to their path metric."""
      ind = np.argsort(self.msg_pm, axis=-1)
      self.msg_pm = np.take_along_axis(self.msg_pm, ind, axis=1)
      self._dec_pointer = np.take_along_axis(self._dec_pointer, ind, axis=1)
  def _cn_op_np(self, x, y):
      """Check node update (boxplus) for LLRs in Numpy.
      See [Stimming_LLR]_ and [Hashemi_SSCL]_ for detailed equations.
      """
      x_in = np.maximum(np.minimum(x, self._llr_max), -self._llr_max)
      y_in = np.maximum(np.minimum(y, self._llr_max), -self._llr_max)
      # avoid division for numerical stability
      llr_out = np.log(1 + np.exp(x_in + y_in))
      llr_out -= np.log(np.exp(x_in) + np.exp(y_in))
      return llr_out
  def _vn_op_np(self, x, y, u_hat):
      """Variable node update (boxplus) for LLRs in Numpy."""
      return np.multiply((1-2*u_hat), x) + y
  def _duplicate_paths_np(self):
      """Copy first ``list_size``/2 paths into lower part in Numpy.
      Decoder indices are encoded in ``self._dec_pointer``.
      """
      ind_low = self._dec_pointer[:, :self._list_size]
      ind_up = self._dec_pointer[:, self._list_size:]
      for i in range(ind_up.shape[0]):
          self.msg_uhat[i, ind_up[i,:], :, :] = self.msg_uhat[i,ind_low[i,:],:, :]
          self.msg_llr[i, ind_up[i,:],:,:] = self.msg_llr[i, ind_low[i,:],:,:]
      # pm must be sorted directly (not accessed via pointer)
      self.msg_pm[:, self._list_size:] = self.msg_pm[:, :self._list_size]
  def _polar_decode_scl_np(self, cw_ind):
      """Recursive decoding function in Numpy.
      We follow the terminology from [Hashemi_SSCL]_ and [Stimming_LLR]_
      and branch the messages into a `left` and `right` update paths until
      reaching a leaf node.
      Tree pruning as proposed in [Hashemi_SSCL]_ is used to minimize the
      tree depth while maintaining the same output.
      """
      n = len(cw_ind)
      stage_ind = int(np.log2(n))
      # recursively branch through decoding tree
      if n>1:
          # prune tree if rate-0 subcode or rep-code is detected
          if self._use_fast_scl:
              if np.sum(self._frozen_ind[cw_ind])==n:
                  # rate0 code detected
                  self._update_rate0_code_np(cw_ind)
                  return
              if (self._frozen_ind[cw_ind[-1]]==0 and
                  np.sum(self._frozen_ind[cw_ind[:-1]])==n-1):
                  # rep code detected
                  self._update_rep_code_np(cw_ind)
                  return
          cw_ind_left = cw_ind[0:int(n/2)]
          cw_ind_right = cw_ind[int(n/2):]
          # ----- left branch -----
          llr_left = self.msg_llr[:, :, stage_ind, cw_ind_left]
          llr_right = self.msg_llr[:, :, stage_ind, cw_ind_right]
          self.msg_llr[:, :, stage_ind-1, cw_ind_left] = self._cn_op_np(
            llr_left,llr_right)
          # call left branch decoder
          self._polar_decode_scl_np(cw_ind_left)
          # ----- right branch -----
          u_hat_left_up = self.msg_uhat[:, :, stage_ind-1, cw_ind_left]
          llr_left = self.msg_llr[:, :, stage_ind, cw_ind_left]
          llr_right = self.msg_llr[:, :, stage_ind, cw_ind_right]
          self.msg_llr[:, :, stage_ind-1, cw_ind_right] = self._vn_op_np(
                llr_left,llr_right,u_hat_left_up)
          # call right branch decoder
          self._polar_decode_scl_np(cw_ind_right)
          # combine u_hat
          u_hat_left_up = self.msg_uhat[:, :, stage_ind-1, cw_ind_left]
          u_hat_right_up = self.msg_uhat[:, :, stage_ind-1, cw_ind_right]
          # u_hat_left_up XOR u_hat_right_up
          u_hat_left =  (u_hat_left_up != u_hat_right_up) + 0
          u_hat = np.concatenate([u_hat_left, u_hat_right_up], axis=-1)
          # provide u_hat for next higher stage
          self.msg_uhat[:, :, stage_ind,  cw_ind] = u_hat
      else: # if leaf is reached perform basic decoding op (=decision)
        self._update_single_bit_np(cw_ind)
        self._update_pm_np(cw_ind)# update PM
        # position is non-frozen
        if self._frozen_ind[cw_ind]==0:
            # sort list
            self._sort_decoders_np()
            # duplicate the best list_size decoders
            self._duplicate_paths_np()
      return
  def _decode_np_batch(self, llr_ch):
    """Decode batch of ``llr_ch`` with Numpy decoder."""
    bs = llr_ch.shape[0]
    # allocate memory for all 2*list_size decoders
    self.msg_uhat = np.zeros([bs,2*self._list_size,self._n_stages+1,self._n])
    self.msg_llr = np.zeros([bs,2*self._list_size,self._n_stages+1,self._n])
    self.msg_pm = np.zeros([bs,2*self._list_size])
    # L-1 decoders start with high penalty
    self.msg_pm[:,1:self._list_size] = self._llr_max
    # same for the second half of the L-1 decoders
    self.msg_pm[:,self._list_size+1:] = self._llr_max
    # use pointers to avoid in-memory sorting
    self._dec_pointer = np.arange(2*self._list_size)
    self._dec_pointer = np.tile(np.expand_dims(self._dec_pointer, axis=0),[bs,1])
    # init llr_ch (broadcast via list dimension)
    self.msg_llr[:, :, self._n_stages, :] = np.expand_dims(llr_ch, axis=1)
    # call recursive graph function
    self._polar_decode_scl_np(self._cw_ind)
    # select most likely candidate
    self._sort_decoders_np()
    # remove pointers
    for ind in range(bs):
        self.msg_uhat[ind, :, :, :] = self.msg_uhat[ind,self._dec_pointer[ind],:, :]
    
    return self.msg_uhat, self.msg_pm
  def _decode_np_hybrid(self, llr_ch, u_hat_sc, crc_valid):
    """Hybrid SCL decoding stage that decodes iff CRC from previous SC
    decoding attempt failed.
    This option avoids the usage of the high-complexity SCL decoder in cases
    where SC would be sufficient. For further details we refer to
    [Cammerer_Hybrid_SCL]_ (we use SC instead of the proposed BP stage).
    Remark: This decoder does not exactly implement SCL as CRC
    can be false positive after the SC stage. However, in these cases
    SCL+CRC may also yield the wrong results.
    Remark 2: Due to the excessive control flow (if/else) and the
    varying batch-sizes, this function is only available as Numpy
    decoder (i.e., runs on the CPU).
    """
    bs = llr_ch.shape[0]
    crc_valid = np.squeeze(crc_valid, axis=-1)
    # index of codewords that need SCL decoding
    ind_invalid = np.arange(bs)[np.invert(crc_valid)]
    # init SCL decoder for bs_hyb samples requiring SCL dec.
    llr_ch_hyb = np.take(llr_ch, ind_invalid, axis=0)
    msg_uhat_hyb, msg_pm_hyb = self._decode_np_batch(llr_ch_hyb)
    # merge results with previously decoded SC results
    msg_uhat = np.zeros([bs, 2*self._list_size, 1, self._n])
    msg_pm = np.ones([bs, 2*self._list_size]) * self._llr_max * self.k
    msg_pm[:, 0] = 0
    # copy SC data
    msg_uhat[:, 0, 0, self._info_pos] = u_hat_sc
    ind_hyb = 0
    for ind in range(bs):
        if not crc_valid[ind]:
            #copy data from SCL
            msg_uhat[ind, :, 0, :] = msg_uhat_hyb[ind_hyb, :, 0, :]
            msg_pm[ind, :] = msg_pm_hyb[ind_hyb, :]
            ind_hyb += 1
    return msg_uhat, msg_pm
  ########################## Keras layer functions#########################
  def build(self, input_shape):
      """Build and check if shape of input is invalid."""
      assert (input_shape[-1]==self._n), "Invalid input shape."
      assert (len(input_shape)>=2), 'Inputs must have at least 2 dimensions.'
  def forward(self,inputs): #inputs (tf.float32): Tensor `[...,n]` containing channel LLR values (as logits).
      """Successive cancellation list (SCL) decoding function.
      This fn performs scl dec and returns estimated info bits.
      An outer CRC can be applied optionally by setting ``crc_degree``.
      Args:
      Returns: `tf.float32`: Tensor `[...,k]` containing hard-decided estimations of all ``k`` information bits.
      Raises:ValueError: If ``inputs`` shape !=`[..., n]` or `dtype` !=`tf.float32`.
          InvalidArgumentError: When rank(``inputs``)<2.
      """
      assert inputs.dtype==self.output_dtype,"Invalid input dtype."
      # internal calculations still in tf.float32
      inputs = inputs.to(tc.float32)
      # last dim must be of length n
      assert inputs.shape[-1] == self._n,"Last input dimension must be of length n."
      # Reshape inputs to [-1, n]
      assert inputs.dim()> 1
      input_shape = inputs.shape
      llr_ch = inputs.reshape([-1, self._n])
      llr_ch = -1. * llr_ch # logits are converted into "true" llrs
      # if activated use Numpy decoder
      
      if self._use_hybrid_sc:
        raise Exception('not implement...') 
      else:
        if self._cpu_only:
          msg_uhat, msg_pm = self._decode_np_batch(llr_ch)
          # restore shape information
          msg_uhat = msg_uhat.reshape([-1, 2*self._list_size, self._n_stages+1, self._n])
          msg_pm = msg_pm.reshape([-1, 2*self._list_size])
        else:raise Exception('not_implement...') 
      # check CRC (and remove CRC parity bits)
      if self._use_crc:
        u_hat_list = msg_uhat[:,:,0,self._info_pos]
        # undo input bit interleaving
        # remark:  output is not interleaved for compatibility with SC
        if self._iil:  
          raise Exception('not_implement...')
        else:   # no interleaving applied
          u_hat_list_crc = u_hat_list  
        _, crc_valid = self._crc_decoder(u_hat_list_crc)
        # add penalty to pm if CRC fails
        pm_penalty = (1.-crc_valid)*self._llr_max*self.k #todo self._k
        msg_pm += np.squeeze(pm_penalty,axis=2)
      # select most likely candidate
      cand_ind =  np.argmin(msg_pm, axis=-1) #(100,)
      #todo
      batch_size = msg_uhat.shape[0]
      c_hat = msg_uhat[np.arange(batch_size), cand_ind, 0, :]
      #* c_hat[100,128] 
      # c_hat = tc.gather(msg_uhat[:, :, 0, :], cand_ind, axis=1, batch_dims=1)
      # u_hat = tc.gather(c_hat, self._info_pos, axis=-1)
      u_hat = c_hat[:,self._info_pos]
      #* u_hat[100,64]
      # and reconstruct input shape
      output_shape = list(input_shape)
      output_shape[-1] = self.k
      output_shape[0] = -1 # first dim can be dynamic (None)
      u_hat_reshape = u_hat.reshape(output_shape)
      if self._return_crc_status:
        raise Exception('not implement...') 
      else: # return only info bits
          return tc.from_numpy(u_hat_reshape).to(self.output_dtype).to(device=self.device)

class Polar5GDecoder(nn.Module):
  """Wrapper for 5G compliant decoding including rate-recovery and CRC removal.
  Note----supports uplink & downlink Polar rate-matching scheme without `codeword segmentation`. 
    Although dec `list size` is not provided by 3GPP [3GPPTS38212]_, 
    people agreed on `list size`=8 for 5G dec reference curves [Bioglio_Design]_.
    All list-decoders apply `CRC-aided` dec"""
  def __init__(self,enc_polar:Polar5GEncoder,#Polar5GEncoder :used for enc including rate-matching.
    dec_type="SC", # 1 of `{"SC", "SCL", "hybSCL", "BP"}`
    list_size=8,return_crc_status=False,#If True, dec also returns CRC status indicating if a codeword was (most likely) correctly recovered.
    output_dtype=tc.float32, 
  ):
    super().__init__()
    self._output_dtype = output_dtype
    self._n_target = enc_polar.n_target; self._k_target = enc_polar.k_target
    self._n_polar = enc_polar.n_polar;   self._k_polar = enc_polar.k_polar
    self._k_crc = enc_polar.enc_crc.crc_length
    self._bil = enc_polar._channel_type == "uplink"
    self._iil = False# enc_polar._channel_type == "downlink"
    
    self._llr_max = 100 # Internal max LLR value (for punctured positions)
    self._enc_polar = enc_polar; self._dec_type = dec_type
    # Initialize the de-interleaver patterns
    self._init_interleavers()
    # Initialize decoder
    if dec_type=="SC":
        print("Warning: CRC cant be used with SC dec and. Please use SCL dec.")
        self._polar_dec = SC_Dec(self._enc_polar._frozen_pos,self._n_polar)
    elif dec_type=="SCL":
      self._polar_dec = SCL_Dec(self._enc_polar._frozen_pos,
          self._n_polar,crc_degree=self._enc_polar.enc_crc.crc_degree,
          list_size=list_size)#, ind_iil_inv = self.ind_iil_inv)
    elif dec_type=="hybSCL":
      self._polar_dec = SCL_Dec(self._enc_polar._frozen_pos,
          self._n_polar, crc_degree=self._enc_polar.enc_crc.crc_degree,
          list_size=list_size, use_hybrid_sc=True, ind_iil_inv = self.ind_iil_inv) 
    else:raise ValueError("Unknown value for dec_type.")
    assert isinstance(return_crc_status, bool),"return_crc_status must be bool."
    self._return_crc_status = return_crc_status
    if self._return_crc_status: # init crc decoder
        if dec_type in ("SCL", "hybSCL"):
            # re-use CRC decoder from list decoder
            self._dec_crc = self._polar_dec._crc_decoder
        else: # init new CRC decoder for BP and SC
            self._dec_crc = CRCDecoder(self._enc_polar._enc_crc)
  def _init_interleavers(self):
    """Initialize inverse interleaver patterns for rate-recovery."""
    # Channel interleaver
    ind_ch_int = self._enc_polar.channel_interleaver(
                                            np.arange(self._n_target))
    self.ind_ch_int_inv = np.argsort(ind_ch_int) # Find inverse perm
    # Sub-block interleaver
    ind_sub_int = self._enc_polar.subblock_interleaving(
                                            np.arange(self._n_polar))
    self.ind_sub_int_inv = np.argsort(ind_sub_int) # Find inverse perm
    # input bit interleaver
    if self._iil:
        self.ind_iil_inv = np.argsort(self._enc_polar.input_interleaver(
                                            np.arange(self._k_polar)))
    else:          self.ind_iil_inv = None
  def forward(self,inputs:tc.Tensor): # inputs`[...,n]`:channel logits/llr values.
    """Polar dec & rate-recovery for uplink 5G Polar codes.
    Returns: b_hat :`[...,k]` hard-decided estimates of all ``k`` information bits.
    crc_status:bool: CRC status indicating if a codeword was (most likely) correctly
    recovered.only returned if `return_crc_status` is True.Note that false positives are possible"""
    inputs = inputs.to(tc.float32) # Reshape inputs to [-1, n]
    input_shape = inputs.shape
    assert len(input_shape)>1
    new_shape = [-1, self._n_target]
    llr_ch = inputs.reshape(new_shape)
    # Note: logits not inverted here; this is done in decoder itself
    # 1.) Undo channel interleaving
    if self._bil:
        llr_deint = llr_ch[:,self.ind_ch_int_inv]  #todo
    else:
        llr_deint = llr_ch
    # 2.) Remove puncturing, shortening, repetition (see Sec. 5.4.1.2)
    # a) Puncturing: set LLRs to 0
    # b) Shortening: set LLRs to infinity
    # c) Repetition: combine LLRs
    if self._n_target >= self._n_polar:
      # Repetition coding
      # Add the last n_rep positions to the first llr positions
      n_rep = self._n_target - self._n_polar
      llr_1 = llr_deint[:,:n_rep]
      llr_2 = llr_deint[:,n_rep:self._n_polar]
      llr_3 = llr_deint[:,self._n_polar:]
      llr_dematched = tc.concat([llr_1+llr_3, llr_2],dim=1)
    else:
      if self._k_polar/self._n_target <= 7/16:
        # Puncturing # Append n_polar - n_target "zero" llrs to first positions
        llr_zero = tc.zeros([llr_deint.shape[0],self._n_polar-self._n_target])
        llr_dematched = tc.concat([llr_zero, llr_deint],dim=1)
      else:
        # Shortening # Append n_polar - n_target "-infinity" llrs to last positions
        # Remark: we still operate with logits here, thus the neg. sign
        llr_infty = -self._llr_max * tc.ones([llr_deint.shape[0],
                                        self._n_polar-self._n_target])
        llr_dematched = tc.concat([llr_deint, llr_infty], dim=1)
    # 3.) Remove subblock interleaving
    llr_dec = llr_dematched[:, self.ind_sub_int_inv]
    # 4.) Run main decoder
    u_hat_crc = self._polar_dec(llr_dec)
    # 5.) Shortening should be implicitly recovered by decoder
    # 6.) Remove input bit interleaving for downlink channels only
    if self._iil:
        u_hat_crc = u_hat_crc[:, self.ind_iil_inv]
    # 7.) Evaluate or remove CRC (and PC)
    if self._return_crc_status:
        # for compatibility with SC/BP, a dedicated CRC decoder is
        # used here (instead of accessing the interal SCL)
        u_hat, crc_status = self._dec_crc(u_hat_crc)
    else: # just remove CRC bits
        u_hat = u_hat_crc[:,:-self._k_crc]
    # And reconstruct input shape
    output_shape = [*input_shape] 
    output_shape[-1] = self._k_target
    output_shape[0] = -1 # First dim can be dynamic (None)
    u_hat_reshape = u_hat.reshape(output_shape)
    # and cast to output dtype
    u_hat_reshape = u_hat_reshape.to(dtype=self._output_dtype)
    if self._return_crc_status:
        # reconstruct CRC shape
        breakpoint()
        output_shape.pop() # remove last dimension
        crc_status = crc_status.reshape(output_shape)
        crc_status = crc_status.to(dtype=self._output_dtype)
        return u_hat_reshape, crc_status
    else:
        return u_hat_reshape