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
    self._nb_stages = int(np.log2(self._n))  #ex. n=8,k=4,_nb_stages=log(8)=3, 3 layers #ex. log(128) = 7
    self._ind_gather,vis_matrix = self._gen_indices(self._n) #(7,129)
    # vis_matrix is just _ind_gather in a easier to understand form
    '''
    ex. n=8, k=4
    _ind_gather 
    [1 8 3 8 5 8 7 8 8] idx0 add with idx1, idx2 add with idx3... 
    [2 3 8 8 6 7 8 8 8] idx0 add with idx2, idx1 add with idx3... 
    [4 5 6 7 8 8 8 8 8] idx0 add with idx4, idx1 add with idx5... 
    vis_matrix  (0 is position of +)
     u0 u2 u3 u4 u5 u6 u7 u8   layer
    [0, 1, 0, 1, 0, 1, 0, 1],  0
    [0, 0, 1, 1, 0, 0, 1, 1],  1
    [0, 0, 0, 0, 1, 1, 1, 1]   2   
    a interesting thing is that inverse of polar generator matrix is equal to itself
    so you can interchange the role of u and c in the diagram below  
    layer 0   1   2      
    u0    +   +   +  c0
    u1        +   +  c1
    u2    +       +  c2
    u3            +  c3
    u4    +   +      c4
    u5        +      c5
    u6    +          c6
    u7               c7
    '''
    b = tc.tensor([[1,1],
                   [0,1]],dtype=tc.float32,device=device)
    m = tc.tensor([[1,1],
                   [0,1]],dtype=tc.float32,device=device)
    for i in range(self._nb_stages-1):
      m = tc.kron(b,m)
    self.G = m
    H1 = self.G[frozen_pos,:]
    H2 = (H1 + 1) %2
    self.H = tc.concatenate((H1,H2),dim=0)
  @property
  def k(self):return self._k #"""Number of information bits."""
  
  def _gen_indices(self, n):
    """Pre-calculate encoding indices stage-wise for tc indexing (gather)."""
    nb_stages = self._nb_stages  # log(n) = log(128) = 7
    # last position denotes empty placeholder (points to element n+1)
    ind_gather = np.ones([nb_stages, n+1]) * n #(7,128+1) default idx is 128(None)
    for s in range(nb_stages):
      ind_range = np.arange(int(n/2))
      ind_dest = ind_range * 2 - np.mod(ind_range, 2**(s))
      ind_origin = ind_dest + 2**s
      ind_gather[s, ind_dest] = ind_origin # and update gather indices
    ind_gather = tc.from_numpy(ind_gather).to(dtype=tc.int32)
    # breakpoint()
    # just for visual  (not used in when encoding)   
    vis_matrix = np.zeros((nb_stages,n),dtype=np.int32)
    for i, row in enumerate(ind_gather):
      for j in row: 
        if int(j)!=n: 
          vis_matrix[i,j] = 1
    # print(vis_matrix)
    return ind_gather, vis_matrix
  def G_matrix(self,bs,c,device='cpu'):
    x_nan = tc.zeros([bs ,1], dtype=self.dtype,device=device) #(bs=100,1)
    x = tc.concat([c, x_nan], dim=1).to(tc.uint8) #(bs,n+1=9) <- (bs,n=8) #append 0 at the end
    #* encode
    for s in range(self._nb_stages):  #7   # loop over all stages
      ind_helper = self._ind_gather[s,:]
      x_add = x[...,ind_helper]
      x = tc.bitwise_xor(x, x_add)
      # print(ind_helper,"\nx      ",x[0],"\nx_add  ",x_add[0],"\nx_final",x_final[0])
    c_out = x[:,0:self._n]  #(bs,n=8) # remove last position  #ex. (bs,n+1=9)->(bs,n=8)
    out = c_out.to(dtype=self.dtype) #uint to `tc.fp32`
    return out
  def forward(self, u):#inputs(tc.fp32):[bs,k] info bits to be encoded.
    """returns polar encoded codewords for given info bits inputs."""
    bs = u.shape[0];  assert u.shape[-1]==self._k,"Last dim must be len k."
    c = tc.zeros([bs,self._n], dtype=self.dtype,device=self.device) #codeword, ex.(bs=100,n=8)
    info_pos_tc = tc.from_numpy(self.info_pos) #ex.n=8,k=4->info_pos=[3,5,6,7] 
    c[...,info_pos_tc] = u#copy info bits to info pos; other pos are frozen (=0)
    # print(u[0],c[0],self.info_pos)
    '''ex. for N=8,k=4  info_pos = [3,5,6,7]
    u = [u0,u1,u2,u3]  
    info_pos=[0,0,0, 1,0, 1, 1, 1]
    c =      [0,0,0,u0,0,u1,u2,u3]
    '''
    out = self.G_matrix(bs,c,device=self.device)  #c, out (bs,n)
    assert tc.sum((self.H@out.T)%2)==0
    # c_hat = self.G_matrix(bs,out) 
    # assert tc.all(c_hat == c)
    return out
  
class Polar5GEncoder(PolarEncoder):
  """5G compliant Polar enc including rate-matching following [3GPPTS38212]_
  for uplink (`UCI`) and downlink (`DCI`). scenario 
  performs polar enc for k info bits and rate-matching so codeword len=n. 
  includes CRC concat & interleave in [3GPPTS38212]_. `block segmentation` not supported (`I_seq=False`).
  We follow basic structure in [Bioglio_Design]_. https://nvlabs.github.io/sionna/api/fec.polar.html
  K -> crc enc -> kpolar=k+kcrc -> channel_interleave(downlink, I_IL) 
    -> kpolar -> subchannel allocation -> npolar -> polar encoding
    -> sub block interleaver -> npolar -> rate matching circular buffer
    -> channel interleaver (uplink ,I_BIL ) -> n
  For further details, we refer to [3GPPTS38212]_, [Bioglio_Design]_ and [Hui_ChannelCoding]_.
  Note---- enc supports (`UCI`)&(`DCI`)  Polar coding from [3GPPTS38212]_.
    For `12 <= k <= 19` 3 additional parity bits in [3GPPTS38212]_ not implement as it would require a
    modified dec procedure to materialize potential gains.
    `Code segmentation` not supported,thus,`n` is limited to max len of 1088 codeword bits.
    For DL scenario, input len limit to `k <= 140` info bits due to limit input bit interleaver size
    [3GPPTS38212]_. For simplicity, we dont exactly re-implement `DCI` scheme from [3GPPTS38212]_. 
    we neglects `all-one` init of CRC shift register and scrambling of CRC parity bits with `RNTI`.
  """
  def __init__(self,k, #int number of info bit per codeword. #* k=64
    n, #int codeword len. #* n = 128
    channel_type="uplink", #Can be "uplink" or "downlink".
    verbose=False, #If True, rate-matching parameters will be printed.
    dtype=tc.float32, #the output datatype of the layer(internal precision remains tf.uint8).
    device='cpu'
  ): #todo  so crc is frozen bits??
    k = int(k) # k or n can be float (e.g. as result of n=k*r)
    n = int(n) # k or n can be float (e.g. as result of n=k*r)
    self.device = device
    assert n>=k, "Invalid coderate (>1).";  assert channel_type in ("uplink","downlink"), "Unsupported channel_type."
    self._channel_type = channel_type
    self._k_target = k; self._n_target = n
    self._verbose = verbose
    # Initialize rate-matcher
    crc_degree, n_polar, frozen_pos, idx_rm, idx_input  = self._init_rate_match(k, n)
    #* k = 64, n= 128 , crc_degree = "CRC11", len(frozen_pos)=53-> k+11+53=k+64
    #?? so crc is not frozen, crc is in info pos
    self._frozen_pos = frozen_pos # Required for decoder
    self._ind_rate_matching = idx_rm # Index for gather-based rate-matching
    self._ind_input_int = idx_input # Index for input interleaver
    super().__init__(frozen_pos, n_polar, dtype=dtype,device=device) # Init super-class (PolarEncoder)
    self._enc_crc = CRCEncoder(crc_degree, k=k,dtype=dtype)# Init CRC encoder
  @property
  def enc_crc(self):return self._enc_crc #CRC encoder layer used for CRC concatenation.
  @property
  def k_target(self):return self._k_target #Number of information bits including rate-matching.
  @property
  def n_target(self):return self._n_target #Codeword length including rate-matching.
  @property
  def k_polar(self):return self._k #Number of information bits of the underlying Polar code.
  @property
  def n_polar(self):return self._n #Codeword length of the underlying Polar code.
  @property
  def k(self):return self._k_target #Number of information bits including rate-matching.
  @property
  def n(self):return self._n_target #Codeword length including rate-matching.
  def subblock_interleaving(self, u):
    #* interleave the puncture frozen position (both uplink downlink use subblock interleave)
    """Input bit interleaving as defined in Sec 5.4.1.1 [3GPPTS38212]_.
    Input-----u: ndarray:1D array to be interleaved. Length of ``u`` must be a multiple of 32.
    Output------: ndarray: Interleaved version of ``u`` with same shape and dtype as ``u``.
    Raises------AssertionError  If length of ``u`` is not a multiple of 32.
    """
    k = u.shape[-1]
    assert np.mod(k,32)==0, "len for sub-block interleaving must be a multiple of 32."
    y = np.zeros_like(u)
    # Permutation according to Tab 5.4.1.1.1-1 in 38.212
    perm = np.array([0, 1, 2, 4, 3, 5, 6, 7, 8, 16, 9, 17, 10, 18, 11, 19,
                     12, 20, 13, 21, 14, 22, 15, 23, 24, 25, 26, 28, 27,
                     29, 30, 31])
    for n in range(k):
      i = int(np.floor(32*n/k))
      j = perm[i] * k/32 + np.mod(n, k/32)
      j = int(j)
      y[n] = u[j]
    return y
  def channel_interleaver(self, c): 
    #* only use at uplink  # downlink dont have channel interleav  
    # I guese using this method we can have different uplink and downlink pattern
    """Triangular interleaver following Sec. 5.4.1.3 in [3GPPTS38212]_.
    Input-----c: ndarray: 1D array to be interleaved.
    Output------: ndarray : Interleaved version of `c` with same shape and dtype as ``c``.
    """
    n = c.shape[-1] # Denoted as E in 38.212
    c_int = np.zeros_like(c)
    # Find smallest T s.t. T*(T+1)/2 >= n
    t = 0
    while t*(t+1)/2 < n: t +=1
    v = np.zeros([t, t])
    ind_k = 0
    for ind_i in range(t):
      for ind_j in range(t-ind_i):
        if ind_k < n:v[ind_i, ind_j] = c[ind_k]
        else: v[ind_i, ind_j] = np.nan # NULL
        # Store nothing otherwise
        ind_k += 1
    ind_k = 0
    for ind_j in range(t):
      for ind_i in range(t-ind_j):
        if not np.isnan(v[ind_i, ind_j]):
          c_int[ind_k] = v[ind_i, ind_j]
          ind_k += 1
    return c_int
  def input_interleaver(self, c): # c:ndarray:1D array to be interleaved.
    """Input interleaver following Sec. 5.4.1.1 in [3GPPTS38212]_.
    Output------:ndarray:Interleaved version of ``c`` with same shape and dtype as ``c``.
    """
    # 38.212 Table 5.3.1.1-1
    p_il_max_table = [0, 2, 4, 7, 9, 14, 19, 20, 24, 25, 26, 28, 31, 34,
      42, 45, 49, 50, 51, 53, 54, 56, 58, 59, 61, 62, 65, 66, 67, 69,
      70, 71, 72, 76, 77, 81, 82, 83, 87, 88, 89, 91, 93, 95, 98, 101,
      104, 106, 108, 110, 111, 113, 115, 118, 119, 120, 122, 123, 126,
      127, 129, 132, 134, 138, 139, 140, 1, 3, 5, 8, 10, 15, 21, 27, 29,
      32, 35, 43, 46, 52, 55, 57, 60, 63, 68, 73, 78, 84, 90, 92, 94, 96,
      99, 102, 105, 107, 109, 112, 114, 116, 121, 124, 128, 130, 133,
      135, 141, 6, 11, 16, 22, 30, 33, 36, 44, 47, 64, 74, 79, 85, 97,
      100, 103, 117, 125, 131, 136, 142, 12, 17, 23, 37, 48, 75, 80, 86,
      137, 143, 13, 18, 38, 144, 39, 145, 40, 146, 41, 147, 148, 149,
      150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162,
      163]
    k_il_max = 164
    k = len(c)
    assert k<=k_il_max, "Input interleaver only defined for length of 164."
    c_apo = np.empty(k, 'int')
    i = 0
    for p_il_max in p_il_max_table:
        if p_il_max >= (k_il_max - k):
            c_apo[i] = c[p_il_max - (k_il_max - k)]
            i += 1
    return c_apo
  ########################## Utility methods#########################
  def _init_rate_match(self, k_target, n_target):  #*ex k_target 12 /n_target 160
    """polar rate matching according to [3GPPTS38212]_.only runs during init 
    For easier alignment with 3GPP prefers `for loop`-based indexing.
    3GPP vs this code: A:k_target;E:n_target; K:k_polar;N:n_polar;L:k_crc
    """
    # Check input for consistency (see Sec. 6.3.1.2.1 for UL)# currently not relevant (segmentation not supported)
    # assert k_target<=1706, "Maximum supported codeword len for Polar is 1706."
    assert n_target >= k_target, "n must be larger or equal k."
    assert n_target >= 18, "n<18 is not supported by the 5G Polar coding scheme."
    assert k_target <= 1013, "k too large - no codeword segmentation supported at the moment."
    assert n_target <= 1088, "n too large - no codeword segmentation supported at the moment."
    # Select CRC polynomials (see Sec. 6.3.1.2.1 (p58)for UL (#*UCI; uplink control info))
    if self._channel_type=="uplink":
      if 12<=k_target<=19: crc_pol = "CRC6"; k_crc = 6
      elif k_target >=20:  crc_pol = "CRC11";k_crc = 11
      else: raise ValueError("k_target<12 is not supported in 5G NR for uplink; please use 'channel coding of small block len' scheme from Sec. 5.3.3 in 3GPP 38.212 instead.")
      n_pc = 0#n_pc_wm = 0# PC bit for k_target=12-19 bits(see Sec.6.3.1.3.1 for UL)
      if k_target<=19:
        n_pc = 0 #n_pc = 3 Currently deactivated
        print("Warning: For 12<=k<=19 additional 3 parity-check bits " \
            "are defined in 38.212. we didn't implement that")
        if n_target-k_target>175:#n_pc_wm = 1 # not implemented
            pass
    else: #* downlink channel
      # for downlink CRC24 is used(in PDCCH msgs are limited to k=140)
      # as input interleaver doesn't support longer seq
      assert k_target <= 140,"k too large for downlink channel config."
      #*  why is 140  because crc is treated as info  -> k_crc = 140+24=164 = K_IL  (downlink)
      assert n_target >= 25, "n too small for downlink channel config with 24 bit CRC."  #1+24=25
      assert n_target <= 576,"n too large for downlink channel configuration." #* 576+24 = 600
      crc_pol = "CRC24C" # following 7.3.2  #p194
      k_crc = 24
      n_pc = 0
    # No input interleaving for uplink needed  #https://nvlabs.github.io/sionna/api/fec.polar.html
    # Calculate Polar payload length (CRC bits are treated as info bits)
    k_polar = k_target + k_crc + n_pc  #*uplink k_polar:12 + 6 + 0 = 18 
    assert k_polar <= n_target, "Device is not expected to be configured with k_polar + k_crc + n_pc > n_target."
    # Select polar mother code length n_polar
    n_min = 5;  n_max = 10 # For uplink; otherwise 9
    #* Select rate-matching scheme following Sec. 5.3.1  #* p14
    if (n_target <= ((9/8) * 2**(np.ceil(np.log2(n_target))-1)) and
      k_polar/n_target < 9/16):  # 9/8 * 128 and   18/160 < 9/16
      n1 = np.ceil(np.log2(n_target))-1  #* 160-> n1 = 7
    else: n1 = np.ceil(np.log2(n_target))  #* 8
    n2 = np.ceil(np.log2(8*k_polar)) #Lower bound such that rate > 1/8  #*n2 = 8
    n_polar = int(2**np.max((np.min([n1, n2, n_max]), n_min)))  #* n_polar  256 n_target 160
    #* Puncturing and shortening as defined in Sec. 5.4.1.1  #* p29
    #* we want to make it from (n_polar:256,18) to (n_target:160,18)
    prefrozen_pos = [] # List containing the pre-frozen indices
    if n_target < n_polar:  # if E<N
      if k_polar/n_target <= 7/16:# K/E <= 7/16  #* Puncturing
        if self._verbose:print("Using puncturing for rate-matching.")
        n_int =  32 * np.ceil((n_polar-n_target) / 32) #* 256-160=96
        int_pattern = self.subblock_interleaving(np.arange(n_int))
        for i in range(n_polar-n_target):  #256-160=96
            # Freeze additional bits
            prefrozen_pos.append(int(int_pattern[i])) #len(int_pattern=96)
        if n_target >= 3*n_polar/4:  #if E>=3N/4
            t = int(np.ceil(3/4*n_polar - n_target/2) - 1)
        else:t = int(np.ceil(9/16*n_polar - n_target/4) - 1) #t=103
        # Extra freezing
        for i in range(t): prefrozen_pos.append(i) #* len=96+103 = 199
      else:#* Shortening ("through" sub-block interleaver)
        if self._verbose: print("Using shortening for rate-matching.")
        n_int =  32 * np.ceil((n_polar) / 32)
        int_pattern = self.subblock_interleaving(np.arange(n_int))
        for i in range(n_target, n_polar):
            prefrozen_pos.append(int_pattern[i])
    # Remove duplicates
    prefrozen_pos = np.unique(prefrozen_pos)  #*len(prefrozen) 199-> 103
    # Find the remaining n_polar - k_polar - |frozen_set|
    # Load full channel ranking
    ch_ranking, _ = generate_5g_ranking(0, n_polar, sort=False)
    # Remove positions that are already frozen by `pre-freezing` stage
    info_cand = np.setdiff1d(ch_ranking, prefrozen_pos, assume_unique=True)
    #* info_cand = 256-prefrozen(103) = 153
    # Identify k_polar most reliable positions from candidate positions
    info_pos = []  #* k_polar = 18
    for i in range(k_polar): info_pos.append(info_cand[-i-1])
    # Sort and create frozen positions for n_polar indices (no shortening)
    info_pos = np.sort(info_pos).astype(int)
    frozen_pos = np.setdiff1d(np.arange(n_polar),info_pos,assume_unique=True)
    # For downlink only: generate input bit interleaver #https://nvlabs.github.io/sionna/api/fec.polar.html
    if self._channel_type=="downlink":
      if self._verbose: print("Using input bit interleaver for downlink.")
      ind_input_int = self.input_interleaver(np.arange(k_polar))
    else: ind_input_int = None
    # Generate tf.gather indices for sub-block interleaver
    ind_sub_int = self.subblock_interleaving(np.arange(n_polar))
    # Rate matching via circular buffer as defined in Sec. 5.4.1.2
    c_int = np.arange(n_polar)
    idx_c_matched = np.zeros([n_target]) #*160
    if n_target >= n_polar:# Repetition coding
      if self._verbose: print("Using repetition coding for rate-matching")
      for ind in range(n_target):
          idx_c_matched[ind] = c_int[np.mod(ind, n_polar)]
    else: #* n_target 160 < n_polar = 256
      if k_polar/n_target <= 7/16:# Puncturing
        for ind in range(n_target): idx_c_matched[ind] = c_int[ind+n_polar-n_target]
        #  96 ~ 255                                             i + 256-160
      else: # Shortening
        for ind in range(n_target): idx_c_matched[ind] = c_int[ind]  #0~96 not this
    # For uplink only: generate input bit interleaver
    if self._channel_type=="uplink":
      if self._verbose:
          print("Using channel interleaver for uplink.")
      ind_channel_int = self.channel_interleaver(np.arange(n_target)) #*160
      # Combine indices for single tf.gather operation
      ind_t = idx_c_matched[ind_channel_int].astype(int)  #* 160
      idx_rate_matched = ind_sub_int[ind_t]
    else: # no channel interleaver for downlink
      idx_rate_matched = ind_sub_int[idx_c_matched.astype(int)]
    if self._verbose:
      print(f"Code params after rate-matching: k = {k_target}, n = {n_target}")
      print(f"Polar mother code: k_polar = {k_polar}, n_polar = {n_polar}")
      print("Using", crc_pol)
      print("Frozen positions: ", frozen_pos)
      print("Channel type: " + self._channel_type)
    return crc_pol, n_polar, frozen_pos, idx_rate_matched, ind_input_int
  def forward(self, u):#inputs(tc.fp32):[bs,k] info bits to be encoded.
    """Polar encoding function including rate-matching and CRC encoding.
    returns polar encoded codewords for given info bits ``inputs`` following [3GPPTS38212]_ including rate-matching.
    Returns:`tf.float32`: Tensor of shape `[...,n]`."""
    bs = u.shape[0];  assert u.shape[-1]==self.k,"Last dim must be len k."
    # Consistency check (i.e., binary) of inputs will be done in super_class
    # CRC encode
    u_crc = self._enc_crc(u)
    # For downlink only: apply input bit interleaver
    if self._channel_type=="downlink": #todo 
      u_crc = tc.gather(u_crc, self._ind_input_int, dim=-1)
      raise Exception('error...') 
    else:
      pass
    # Encode bits (= channel allocation + Polar transform)
    c = super().forward(u_crc)
    #* (bs,18) -> (bs,256)
    # Sub-block interleaving with 32 sub-blocks as in Sec. 5.4.1.1
    # Rate matching via circular buffer as defined in Sec. 5.4.1.2
    # For uplink only: channel interleaving (i_bil=True)
    c_matched = c[:,self._ind_rate_matching]
    #c (1000,256) [:,ind_rate_match (160,)]-> c_match(1000,160)
    # Restore original shape
    input_shape_list = list(u.shape)
    output_shape = input_shape_list[0:-1] + [self._n_target]
    output_shape[0] = -1 # To support dynamic shapes
    c_reshaped = c_matched.reshape(output_shape)
    #* (1000,160)
    return c_reshaped