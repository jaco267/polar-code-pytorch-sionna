import numpy as np
import torch as tc
from torch import nn
import scipy as sp # for sparse H matrix computations
import itertools
from my_sn.fec.utils import int_mod_2
from my_sn.sim import hard_decisions
class OSDecoder(nn.Module):
  r"""Ordered statistics decoding (OSD) for binary, linear block codes.
  OSD algo proposed in [Fossorier]_ ,approximates ml dec for a sufficiently large order :`t`. 
  works for arbitrary linear block codes, but has high complexity for long codes.
  The algo consists of following steps:
  1. Sort LLRs according to their reliability & apply same col permutation to G matrix.
  2. Bring permuted G matrix into its systematic form(so-called *most-reliable basis*).
  3. Hard-decide & re-encode `k` most reliable bits & discard remaining `n-k` received pos.
  4. Gen all possible err patterns up to `t` errors in `k` most reliable pos find most likely codeword within these candidates.
  we uses LLR-based distance metric from [Stimming_LLR_OSD]_ which simplifies handling of higher-order modu schemes.
  Note----
  OS dec is high complex and is only for small `t` as `{n \choose t}` patterns must be evaluated. 
  OSD works for any linear block codes & provides an est of ML perf for sufficiently large `t`. 
  """
  def __init__(self,t=0, #int Order of OSD algo
    encoder=None, #Layer: FEC encoder. code used to init OSD.
    dtype=tc.float32,
    device='cpu'):
    self.device = device
    self.dtype=  dtype
    super().__init__()
    self._llr_max = 100. # internal clipping value for llrs
    if dtype not in (tc.float16, tc.float32, tc.float64):
        raise ValueError('dtype must be {tf.float16, tf.float32, tf.float64}.')
    assert (int(t)==t), "t must be int."
    self._t = int(t)##"""Order of the OSD algorithm"""
    # test that encoder is already initialized (relevant for conv codes)
    if encoder.k is None:raise AttributeError("It seems as if encoder is not init or has no attribute k.")
    # Encode identity matrix to get k basis vectors of the code
    
    u = tc.eye(encoder.k,device= device)  # Shape: (k, k) 
    # Encode and remove batch dimension
    self._gm = encoder(u).to(dtype)  # Shape: (k, n)#"""Generator matrix of the code"""
    self._k = self._gm.size(0)##"""Number of information bits per codeword"""
    self._n = self._gm.size(1)#"""Codeword length"""
    # Initialize error patterns
    num_patterns = self._num_error_patterns(self._n, t)
    # Storage/computational complexity scales with n
    num_symbols = num_patterns * self._n
    if num_symbols > 1e9:  # number still to be optimized
        print(f"Note: Required memory complexity is large for given code params and t={t}. Please consider small batch-sizes")
    if num_symbols > 1e11:raise ResourceWarning("OSD cant run this (complexity too high). Please use a smaller value for t.")
    # Pre-compute all error patterns
    self._err_patterns = []
    for t_i in range(1, t + 1):
        self._err_patterns.append(self._gen_error_patterns(self._k, t_i))
  ######## Utility methods #########################
  def _num_error_patterns(self, n, #length of vector. 
    t  #number of errors.
  ):  #* done
    r"""get num of possible err patterns for t errors in n, pos, i.e., get `{n \choose t}`."""
    return sp.special.comb(n, t, exact=True, repetition=False)
  def _gen_error_patterns(self, n, #Length of vector.
    t  #Number of errors.
  ):#Output------: [num_patterns, t], size`num_patterns`=`{n \choose t}` containing t error indices.
    r"""Returns list of all possible error patterns for t errs in n pos."""
    err_patterns = []
    for p in itertools.combinations(range(n), t):
        err_patterns.append(p)
    return err_patterns
  def _get_dist(self,llr,#[bs, n], fp32 .Received llrs of channel observations.
    c_hat #[bs, num_cand, n], fp32 Candidate codewords for which distance to ``llr`` shall be evaluated.
  ):
    """Distance fn used for ML candidate selection.
    Currently, distance metric from Polar decoding [Stimming_LLR_OSD]_ literature is implemented. 
    Output------: [bs, num_cand],Distance between ``llr`` & ``c_hat`` for each of `num_cand` codeword candidates.
    Reference---------[Stimming_LLR_OSD] Alexios Balatsoukas-Stimming, Mani Bastani Parizi,
    Andreas Burg, "LLR-Based SCL Decoding of Polar Codes." IEEE Trans Signal Processing, 2015.
    """
    # broadcast llr to all codeword candidates
    llr = tc.unsqueeze(llr,dim=1)
    llr_sign = llr * (-2.*c_hat + 1.) # apply BPSK mapping
    d= tc.log(1.+tc.exp(llr_sign))
    return tc.mean(d,dim=2)
  def _find_min_dist(self, llr_ch,  #[bs, n], Channel observations as llrs after mrb sorting.
    ep,     #[num_patterns, t],`num_patterns`=:math:`{n \choose t}` contain t error indices.
    gm_mrb, # [bs, k, n] Most reliable basis for each batch example.
    c       #[bs, n], Most reliable base codeword.
  ):
    r"""Find error pattern which leads to min distance.
    Output------
    : [bs]:fp32: Distance of most likely codeword to ``llr_ch`` after testing all ``ep`` error patterns.
    : [bs,n]: fp32: most likely codeword after testing against all ``ep`` error patterns.
    """
    # gen all test candidates for each possible error pattern
    e = gm_mrb[:,ep]   
    e = tc.sum(e,dim=2)
    e += tc.unsqueeze(c, dim=1) #add to mrb codeword  
    c_cand = int_mod_2(e)# apply modulo-2 operation
    # calculate distance for each candidate
    # where c_cand has shape [bs, num_patterns, n]
    d = self._get_dist(llr_ch,c_cand)
    # find candidate index with smallest metric  
    idx = tc.argmin(d,dim=1) 
    bsss = c_cand.shape[0]
    c_hat = c_cand[tc.arange(bsss),idx]
    bsss = d.shape[0]
    d = d[tc.arange(bsss),idx]
    return d, c_hat 
  def _find_mrb(self,gm):
    """Find most reliable basis for all G matrices in batch.
    Output------
    gm_mrb:[bs,k,n] float:Most reliable basis in systematic form for each batch example.
    idx_sort: [bs, n] int: Indices of col permutations applied during mrb calculation.
    """
    bs = gm.shape[0]; s = gm.shape  
    idx_pivot =  [] #  bring gm in systematic form (by so-called pivot method)
    for idx_c in range(self._k):
      idx_p = tc.argmax(gm[:,idx_c,:],dim=-1)# find pivot (i.e., first pos with index 1)
      idx_pivot.append(idx_p)# store pivot position
      bsss = gm.shape[0]
      r = gm[tc.arange(bsss),...,idx_p] # eliminate column in all other rows
      # ignore idx_c row itself by adding all-zero row
      rz = tc.zeros((bs,1),dtype= self.dtype,device=self.device)
      r = tc.concat((r[:,:idx_c], rz , r[:,idx_c+1:]),dim=1)
      # mask=0 at all rows where pivot position of this row is zero
      mask = tc.tile(r.unsqueeze(-1),(1,1,self._n))   
      gm_off = gm[:,idx_c,:].unsqueeze(1)  
      # update all row in parallel   
      gm = int_mod_2(gm+mask*gm_off) # account for binary operations
    idx_pivot = tc.stack(idx_pivot).T # pivot positions
    # find non-pivot pos (i.e., all idx that're not part of idx_pivot)
    # add large offset to pivot indices & sorting gives indices of interest
    idx_range =tc.tile(tc.unsqueeze(
       tc.arange(self._n,dtype=tc.int64),dim=0),(bs,1)).to(self.device)
    # large value to be added to irrelevant indices
    updates = self._n * tc.ones((bs,self._k),dtype=tc.int64,device=self.device)
    idx_updates = idx_pivot##idx_updates (3,64,2)
    # add large value to pivot positions  
    idx = idx_range.clone().to(self.device) 
    #idx (3,128)   idx_updates (3,64) #updates(3,64)
    idx.scatter_add_(1, idx_updates, updates) 
    # sort & slice first n-k indices (equals parity pos)
    idx_parity = tc.argsort(idx)[:,:self._n-self._k].to(tc.int64)
    idx_sort = tc.concat((idx_pivot, idx_parity),dim=1)#* idx_sort (3,128)
    # permute gm according to indices idx_sort
    bsss = gm.shape[0]
    gm = gm[tc.arange(bsss).reshape(-1,1),:,idx_sort]
    gm = tc.permute(gm,(0,2,1))#* gm.shape (3,64,128)
    return gm,idx_sort
  def forward(self,inputs): #llrs_ch: [...,n], channel logits/llr values.
    #Output------: [...,n], binary hard-decisions of all codeword bits.
    #"""Applies osd to inputs.llr = p(x=1)/p(x=0)."""
    # flatten batch-dim  
    input_shape = inputs.shape; llr_ch = inputs.reshape(-1,self._n).to(self.dtype)
    bs = llr_ch.shape[0]   
    llr_ch = tc.clip(llr_ch,min=-self._llr_max,max=self._llr_max)# clip inputs  
    # step 1 : sort LLRs  
    idx_sort = tc.argsort(tc.abs(llr_ch),descending=True) #* idx_sort(3,128)
    # permute gm per batch sample individually
    gm = self._gm.unsqueeze(0).expand(bs, self._k, self._n)
    bss = gm.shape[0] #* gm (3,64,128)
    gm_sort = gm[tc.arange(bss).reshape(-1,1),:,idx_sort.reshape(bss,-1)]   
    gm_sort = tc.permute(gm_sort,(0,2,1)) #* gm_sort(3,64,128)
    #step2 : Find most reliable basis (MRB)
    gm_mrb, idx_mrb = self._find_mrb(gm_sort)
    # apply corresponding mrb permutations
    bsss = idx_sort.shape[0]
    idx_sort = idx_sort[tc.arange(bsss).reshape(-1,1),idx_mrb]    
    llr_sort = llr_ch[tc.arange(bsss).reshape(-1,1),idx_sort]
    # find inverse permutation for final output
    idx_sort_inv = tc.argsort(idx_sort) #
    # hard-decide k most reliable positions and encode
    u_hd = hard_decisions(llr_sort[:,0:self._k])
    u_hd = tc.unsqueeze(u_hd,dim=1)
    c = tc.squeeze(tc.matmul(u_hd,gm_mrb),dim=1)#todo tf.matmull
    c = int_mod_2(c)
    # and search for most likely pattern
    # _get_dist expects a list of candidates, thus expand_dims to [bs, 1, n]
    d_best = self._get_dist(llr_sort,tc.unsqueeze(c,dim=1))
    d_best = tc.squeeze(d_best,dim=1)
    c_hat_best = c  
    # known in advance - can be unrolled
    for ep in self._err_patterns:
      # compute distance for all candidate codewords
      d, c_hat =self._find_min_dist(llr_sort,ep,gm_mrb,c)
      # select most likely candidate
      ind = tc.unsqueeze(d<d_best,dim=1)
      c_hat_best = tc.where(ind,c_hat,c_hat_best)
      d_best = tc.where(d<d_best,d,d_best)  
    # undo permutations for final codeword 
    bss = c_hat_best.shape[0]
    c_hat_best = c_hat_best[tc.arange(bss).reshape(-1,1),idx_sort_inv]
    c_hat = c_hat_best.reshape(input_shape)
    return c_hat
