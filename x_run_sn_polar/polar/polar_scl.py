import numpy as np
import torch as tc
from torch import nn
count=  0
class SCL_Dec(nn.Module):
  """PolarSCLDecoder(frozen_pos, n, list_size=8, crc_degree=None, use_hybrid_sc=False, use_fast_scl=True, cpu_only=False, use_scatter=False, ind_iil_inv=None, return_crc_status=False, output_dtype=tf.float32, **kwargs)
  SCL decoder [Tal_SCL]_ for Polar codes & Polar-like codes.
  Input-----[...,n],channel LLR values (as logits).
  Output--b_hat:[...,k],fp32: hard-decided est of `k` info bits.
  Note: SCL dec in [Tal_SCL]_,uses LLR-based msg updates[Stimming_LLR]_. 
    follows notation from [Gross_Fast_SCL]_, [Hashemi_SSCL]_. 
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
    self.device = device; 
    if output_dtype not in (tc.float16, tc.float32, tc.float64):raise ValueError('output_dtype must be {tf.float16, tf.float32, tf.float64}.')
    self.output_dtype = output_dtype# tc.fp32.
    n = int(n) # n can be float (e.g. as result of n=k*r)
    assert len(frozen_pos)<=n, "Num. of elements in frozen_pos cannot be greater than n."
    assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."
    assert np.log2(list_size)==int(np.log2(list_size)), "list_size must be a power of 2."
    # store internal attributes
    self._n = n; self._frozen_pos = frozen_pos
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
  @property
  def n(self):return self._n #"""Codeword length."""
  @property
  def k(self):return self._k #"""Number of information bits."""
  @property
  def frozen_pos(self):return self._frozen_pos #"""Frozen positions for Polar decoding."""
  def _update_single_bit_np(self, ind_u):
    """Update single bit at pos ``ind_u`` of all dec list 2*8
    u0        0
             / \
    u1      o   o   go from u0 to u1   
    """
    #*  ind_u  len always == 1  if len(ind_u) != 1:breakpoint()
    if self._frozen_ind[ind_u]==0: # position is non-frozen
      ind_dec = np.expand_dims(self._dec_pointer[:, self._list_size:],axis=-1)
      #* ind_dec(bs,ls=8,1)  #[8~15], [8~15],..    self._dec_pointer:(1000,16)
      uhat_slice = self.msg_uhat[:, :, 0, ind_u]
      #* uhat_slice:(1000,16,1)   self.msg_uhat_ [bs,ls*2=16,nt+1=7,n=64]
      #* put 1 in uhat_slice  in ind ind_dec   
      #ex u_hat[0]=[0,0..0] become [0,0..0,1..1]  8 1's  [8~15] become 1's 
      # breakpoint()
      np.put_along_axis(uhat_slice, ind_dec, 1., axis=1)
      # breakpoint()
      #todo I got it scl, first 8 list is if u_hat=1 latter 8 list is when u_hat=0, and we calculate path metric 
      self.msg_uhat[:, :, 0, ind_u] = uhat_slice
      #* msg_uhat_ (bs,ls*2=16,n_st+1=7,n=64)   [8~15] in ls*2 all set to 1
  def _update_pm_np(self, ind_u): #*ind_u len=1
    """ Update path metric of all dec at bit pos `ind_u` in Numpy.
    We apply Eq. (10) from [Stimming_LLR]_."""
    #*path metric  lower is better  #https://marshallcomm.cn/2017/03/15/polar-code-7-scl-decoder/
    # assert len(ind_u)==1
    ind = np.expand_dims(self._dec_pointer, axis=-1)
    #* dec_ptr (bs,l_s*2=16), ind(bs,ls*2,1)
    u_hat = np.take_along_axis(self.msg_uhat[:, :, 0, ind_u], ind, axis=1)
    #*u_hat (bs=1000,16),   msg_uhat_ (bs,ls*2=16,n_st+1=7, n=64)
    u_hat = np.squeeze(u_hat, axis=-1)
    llr_in = np.take_along_axis(self.msg_llr[:, :, 0, ind_u], ind, axis=1)
    #*llr_in(bs=1000,16)   msg_llr (bs,ls*2=16,n_st+1=7,n=64)
    llr_in = np.squeeze(llr_in, axis=-1)
    llr_clip = np.maximum(np.minimum(llr_in, self._llr_max), -self._llr_max)
    self.msg_pm += np.log(1 + np.exp(-np.multiply((1-2*u_hat), llr_clip)))
    # breakpoint()
    #* msg_pm_ (bs=1000,16)      add the path metric of this bit
  def _sort_decoders_np(self):
    """Sort dec according to their path metric."""
    ind = np.argsort(self.msg_pm, axis=-1) #* ind (bs,ls*2=16)  sort by path metric
    #* sort from 0~ls*2=16
    self.msg_pm = np.take_along_axis(self.msg_pm, ind, axis=1)# pathmatrix value small~large
    self._dec_pointer = np.take_along_axis(self._dec_pointer, ind, axis=1)
    #* [0,1,2,3,..16]->[8,0,9,...15,...7] #path_metric from small to large
  def _cn_op_np(self, x, y):         #*f func
    """Check node update (boxplus) for LLRs 
    See [Stimming_LLR]_ and [Hashemi_SSCL]_ for detailed equations."""
    x_in = np.maximum(np.minimum(x, self._llr_max), -self._llr_max)
    y_in = np.maximum(np.minimum(y, self._llr_max), -self._llr_max)
    # avoid division for numerical stability
    # llr_out = np.log(1 + np.exp(x_in + y_in))
    # llr_out -= np.log(np.exp(x_in) + np.exp(y_in))
    #max  
    x_tc = tc.from_numpy(x_in)
    y_tc = tc.from_numpy(y_in)
    llr_out = tc.sign(x_tc)*tc.sign(y_tc)*tc.min(tc.abs(x_tc),tc.abs(y_tc))
    llr_out = llr_out.numpy()
    return llr_out
  def _vn_op_np(self, x, y, u_hat):  #*g func
    return np.multiply((1-2*u_hat), x) + y #Variable node update (boxplus) for LLRs
  def _duplicate_paths_np(self):
    """Copy 1st ``list_size``/2 paths into lower part 
    Decoder indices are enc in ``self._dec_pointer``."""
    ind_low = self._dec_pointer[:, :self._list_size]
    #* ind_low (bs,ls=8)    lower path metric  
    ind_up = self._dec_pointer[:, self._list_size:]
    #* ind_up   (bs,ls=8)   higher path metric
    for i in range(ind_up.shape[0]):
        self.msg_uhat[i, ind_up[i,:], :, :] = self.msg_uhat[i,ind_low[i,:],:, :]
        self.msg_llr[i, ind_up[i,:],:,:] = self.msg_llr[i, ind_low[i,:],:,:]
    # pm must be sorted directly (not accessed via pointer)
    self.msg_pm[:, self._list_size:] = self.msg_pm[:, :self._list_size]
  def _polar_decode_scl_np(self, cw_ind):#cw_ind[0~63]
    """Rec dec fn. term from [Hashemi_SSCL]_ & [Stimming_LLR]_ &
    branch msgs into a `left`&`right` update paths until reaching a leaf node.
    Tree pruning as proposed in [Hashemi_SSCL]_ is used to minimize tree depth while maintaining same output.
    paper:Simplified SCL Decoding of Polar Code"""
    n = len(cw_ind) #* n = 64, msg_llr_[0,0,:,0]=[0,..0,0.6], msg_uhat_[0,0,:,0]=[0,...,0]
    stage_ind = int(np.log2(n)) #* stage_ind=6
    global count
    # recursively branch through decoding tree
    if n>1:  #todo use_fast_scl
      cw_ind_left = cw_ind[0:int(n/2)] #*(n//2=32,)[0~31]
      cw_ind_right = cw_ind[int(n/2):] #*(n//2=32,)[32~63]
      # ----- left branch ----- #msg_llr_(bs,ls*2=16,n_st+1=7,n=64)
      llr_left = self.msg_llr[:, :, stage_ind, cw_ind_left]  #stage=6 llr_left(bs,ls*2=16,n/2=32)
      llr_right = self.msg_llr[:, :, stage_ind, cw_ind_right]#stage=6 llr_right(bs,ls*2=16,n/2=32)
      f_val = self._cn_op_np(llr_left,llr_right)#f_func
      self.msg_llr[:,:,stage_ind-1, cw_ind_left]=f_val#*msg_llr_(bs,ls*2,n_st=7,n=64)  
      self._polar_decode_scl_np(cw_ind_left) #left branch decoder --- # print(np.all(llr_right==self.msg_llr[:, :, stage_ind, cw_ind_right]))
      # ----- right branch -----  # if stage_ind-1==6:breakpoint()  #* last ind of msg_uhat_ is always unused #todo why msg_uhat_ is stage_ind-1??
      u_hat_left_up = self.msg_uhat[:, :, stage_ind-1, cw_ind_left]   # if not np.all(llr_right==self.msg_llr[:, :, stage_ind, cw_ind_right]): breakpoint() #  2nd dim is different (ls)#*  not equal is because  _sort_decoders_np_ or duplicate_path_np
      llr_left = self.msg_llr[:, :, stage_ind, cw_ind_left] #* msg_llr have been update
      llr_right = self.msg_llr[:, :, stage_ind, cw_ind_right]
      g_val = self._vn_op_np(llr_left,llr_right,u_hat_left_up)
      self.msg_llr[:, :, stage_ind-1, cw_ind_right] = g_val
      self._polar_decode_scl_np(cw_ind_right)# call right branch decoder
      #*---- reencode----
      u_hat_left_up = self.msg_uhat[:, :, stage_ind-1, cw_ind_left]
      u_hat_right_up = self.msg_uhat[:, :, stage_ind-1, cw_ind_right]
      # u_hat_left_up XOR u_hat_right_up
      u_hat_left =  (u_hat_left_up != u_hat_right_up) + 0
      u_hat = np.concatenate([u_hat_left, u_hat_right_up], axis=-1)
      # provide u_hat for next higher stage
      self.msg_uhat[:, :, stage_ind,  cw_ind] = u_hat
      # print(f"---{count=}---n----pm:{self.msg_pm}",n,self._dec_pointer,
      #         "\n",tc.tensor(self.msg_uhat[0][0][0]))
      # print(u_hat_left_up.reshape(-1))
      # print(u_hat_right_up.reshape(-1))
      # print(u_hat_left.reshape(-1)) #*n
      # print(u_hat.reshape(-1)) #* 2*n
      # if count>=8:breakpoint()
      count+=1
      
      # if np.sum(self.msg_uhat)>0:breakpoint()
    else: # if leaf is reached perform basic decoding op (=decision)
      # if np.sum(self.msg_uhat)>0:breakpoint()
      # print(f"----{count=}-----!!")
      # if count>=8:breakpoint()
      self._update_single_bit_np(cw_ind)  #* almost
      self._update_pm_np(cw_ind)# update PM  #* almost
      # if np.sum(self.msg_uhat)>0:breakpoint()
      #todo where is the case when frozen_ind ==1 ??
      if self._frozen_ind[cw_ind]==0:# pos is non-frozen
        # if np.sum(self.msg_uhat)>0:breakpoint()
        self._sort_decoders_np()# sort list  #*almost
        self._duplicate_paths_np()# duplicate best list_size decoders
        # if np.sum(self.msg_uhat)>0:breakpoint()
    return
  def _decode_np_batch(self, llr_ch):
    """Decode batch of ``llr_ch`` with Numpy decoder."""
    bs = llr_ch.shape[0]
    '''ex. n = 8  3 stage
    s0 s1 s2 s3        s0 s1 s2 s3
    msg_llr            msg_uhat_
    0        llr[0]    0
    ...                ...
    0                  0
    0        llr[7]    0  
    '''
    # allocate memory for all 2*list_size decoders
    self.msg_uhat = np.zeros([bs,2*self._list_size,self._n_stages+1,self._n])
    self.msg_llr = np.zeros([bs,2*self._list_size,self._n_stages+1,self._n])
    self.msg_pm = np.zeros([bs,2*self._list_size])
    self.msg_pm[:,1:self._list_size] = self._llr_max # L-1 decoders start with high penalty
    self.msg_pm[:,self._list_size+1:] = self._llr_max# same for 2nd half of L-1 dec
    #* 0 30 30,...30  0 30 30 ...30
    self._dec_pointer = np.arange(2*self._list_size)# use ptrs to avoid in-memory sorting
    self._dec_pointer = np.tile(np.expand_dims(self._dec_pointer, axis=0),[bs,1])
    #* (bs,16) [0,  1,  2, ..., 13, 14, 15]
    # init llr_ch (broadcast via list dimension)
    self.msg_llr[:, :, self._n_stages, :] = np.expand_dims(llr_ch, axis=1)
    #* msg_llr (bs,list_size*2=16,n_stage+1=7,n=64)      llr_ch (bs,1,n=64)
    # call recursive graph function
    self._polar_decode_scl_np(self._cw_ind)#cw_ind = [0,1,...63]
    self._sort_decoders_np()# select most likely candidate
    for ind in range(bs):
        self.msg_uhat[ind, :, :, :] = self.msg_uhat[ind,self._dec_pointer[ind],:, :]
    #* msg_pm is different 
    # breakpoint()
    return self.msg_uhat, self.msg_pm
  def forward(self,inputs): #inputs (tf.float32): Tensor `[...,n]` containing channel LLR values (as logits).
    """performs scl dec & returns est info bits.
    Args: Returns: `[...,k]` containing hard-decided est of all `k` info bits."""
    assert inputs.dtype==self.output_dtype,"Invalid input dtype."
    inputs = inputs.to(tc.float32)
    assert inputs.shape[-1] == self._n,"Last input dim must be of len n."
    assert inputs.dim()> 1
    input_shape = inputs.shape  #* (bs,n=64)
    llr_ch = inputs.reshape([-1, self._n])#* (bs,n=64)
    llr_ch = -1. * llr_ch # logits are converted into "true" llrs
    msg_uhat, msg_pm = self._decode_np_batch(llr_ch)
    #*msg_uhat (bs,list_size*2=16,n_stages+1=6+1,n=64)  #msg_pm (bs,list_size*2=16)
    #* 2**n_stage = 2**6 = 64 = n
    # select most likely candidate
    cand_ind =  np.argmin(msg_pm, axis=-1) #(100,)
    batch_size = msg_uhat.shape[0]
    c_hat = msg_uhat[np.arange(batch_size), cand_ind, 0, :] #* c_hat[100,128] 
    # breakpoint()
    u_hat = c_hat[:,self._info_pos] #* u_hat[100,64]
    # and reconstruct input shape
    output_shape = list(input_shape)
    output_shape[-1] = self.k
    output_shape[0] = -1 # 1st dim can be dynamic (None)
    u_hat_reshape = u_hat.reshape(output_shape)
    return tc.from_numpy(u_hat_reshape).to(self.output_dtype).to(device=self.device)
