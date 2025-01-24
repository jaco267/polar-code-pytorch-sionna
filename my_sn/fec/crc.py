import numpy as np
import torch as tc
from torch import nn
from my_sn.fec.utils import int_mod_2
device = 'cuda' if tc.cuda.is_available() else 'cpu'
class CRCEncoder(nn.Module):
  """Adds cyclic redundancy check to input sequence.
  CRC polynomials from Sec. 5.1 in [3GPPTS38212_CRC]_ are available:
  `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.
  Note----For performance enhance, we use a G-matrix based
    algo for fixed `k` instead of more common shift register-based operations. 
    Thus, enc need to trigger an (internal) rebuild if `k` changes.
  """
  def __init__(self, 
    crc_degree, #Defining CRC polynomial to be used. one of `{CRC24A, CRC24B, CRC24C, CRC16, CRC11, CRC6}`.
    k,dtype=tc.float32 #the output dtype.
  ):
    super().__init__()
    assert isinstance(crc_degree, str), "crc_degree must be str"
    self.dtype=dtype
    self._crc_degree = crc_degree
    # init 5G CRC polynomial
    self._crc_pol, self._crc_length = self._select_crc_pol(self._crc_degree)
    self._k = k; self._n = None
    self.build([None,k]) #64
  ######## Public methods and properties########
  @property
  def crc_degree(self):return self._crc_degree #CRC degree as string
  @property
  def crc_length(self):return self._crc_length #Length of CRC. Equals number of CRC parity bits.
  @property
  def crc_pol(self):return self._crc_pol #CRC polynomial in binary representation.
  @property
  def k(self):return self._k #Number of information bits per codeword
  @property
  def n(self):return self._n #Number of codeword bits
  ################# Utility methods####################
  def _select_crc_pol(self, crc_degree):
    """Select 5G CRC polynomial according to Sec. 5.1 [3GPPTS38212_CRC]_."""
    if crc_degree=="CRC24A":  crc_length = 24; crc_coeffs = [24, 23, 18, 17, 14, 11, 10, 7, 6, 5, 4, 3, 1, 0]
    elif crc_degree=="CRC24B":crc_length = 24; crc_coeffs = [24, 23, 6, 5, 1, 0]
    elif crc_degree=="CRC24C":crc_length = 24; crc_coeffs = [24, 23, 21, 20, 17, 15, 13, 12, 8, 4, 2, 1, 0]
    elif crc_degree=="CRC16": crc_length = 16; crc_coeffs = [16, 12, 5, 0]
    elif crc_degree=="CRC11": crc_length = 11; crc_coeffs = [11, 10, 9, 5, 0]
    elif crc_degree=="CRC6":  crc_length = 6;  crc_coeffs = [6, 5, 0]
    else:raise ValueError("Invalid CRC Polynomial")
    crc_pol = np.zeros(crc_length+1)
    for c in crc_coeffs: crc_pol[c] = 1
    # invert array (MSB instead of LSB)
    crc_pol_inv = np.zeros(crc_length+1)
    for i in range(crc_length+1):
        crc_pol_inv[crc_length-i] = crc_pol[i]
    return crc_pol_inv.astype(int), crc_length
  def _gen_crc_mat(self, k, pol_crc):
    """ Build (dense) G matrix for CRC parity bits. treat CRC as systematic linear code, i.e.,
    G matrix can be composed out of `k` linear ind(valid) codewords. 
    For this, we CRC encode all `k` unit-vectors`[0,...1,...,0]` and build G matrix.
    To avoid `O(k^2)` complexity, we start with last unit vector
    given as `[0,...,0,1]` and can gen result for next vector
    `[0,...,1,0]` via another polynom division of remainder from previous result. 
    This allows to successively build G matrix at complexity `O(k)`.
    """
    crc_length = len(pol_crc) - 1
    g_mat = np.zeros([k, crc_length])
    x_crc = np.zeros(crc_length).astype(int)
    x_crc[0] = 1
    for i in range(k):
        # shift by one position
        x_crc = np.concatenate([x_crc, [0]])
        if x_crc[0]==1:
            x_crc = np.bitwise_xor(x_crc, pol_crc)
        x_crc = x_crc[1:]
        g_mat[k-i-1,:] = x_crc
    return g_mat
  ########################## Keras layer functions#########################
  def build(self, input_shape):
    """Build G matrix, CRC always added to last dim of input."""
    k = input_shape[-1] # we perform the CRC check on the last dimension
    assert k is not None, "Shape of last dimension cannot be None."
    g_mat_crc = self._gen_crc_mat(k, self.crc_pol)
    self._g_mat_crc = tc.from_numpy(g_mat_crc).to(dtype=tc.float32).to(self.device)
    self._k = k
    self._n = k + g_mat_crc.shape[1]
  def forward(self, inputs): #inputs : [...,k] . rank>= 2.
    #output [...,k+crc_degree], CRC enc bits,same shape as `inputs` except last dim becomes `[...,k+crc_degree]`.
    """cyclic redundancy check function.
    This function add the CRC parity bits to ``inputs`` and returns the
    result of the CRC validation.
    """
    # assert rank must be two
    assert len(inputs.shape)> 1
    # re-init if shape has changed, update generator matrix
    if inputs.shape[-1] != self._g_mat_crc.shape[0]:
        breakpoint()
        self.build(inputs.shape)
    # note: as the code is systematic, we only encode the crc positions
    # this the generator matrix is non-sparse and a "full" matrix
    # multiplication is probably the fastest implementation.
    x_exp = tc.unsqueeze(inputs,dim=-2) # row vector of shape 1xk  #todo expand dim
    #* (bs,1,k=12)
    # tf.matmul onl supports floats (and int32 but not uint8 etc.)
    x_exp32 = x_exp.to(dtype=tc.float32)
    x_crc = x_exp32@ self._g_mat_crc # calculate crc bits
    # take modulo 2 of x_crc (bitwise operations instead of tf.mod)
    x_crc = int_mod_2(x_crc)
    x_crc = x_crc.to(dtype=self.dtype)
    x_conc = tc.concat([x_exp, x_crc], -1) #todo
    x_out = tc.squeeze(x_conc, dim=-2) #todo
    return x_out

class CRCDecoder(nn.Module):
  def __init__(self, crc_encoder,
    dtype=tc.float32,
  ):
    super().__init__()
    assert isinstance(crc_encoder, CRCEncoder), "crc_encoder must be an instance of CRCEncoder."
    self._encoder = crc_encoder#"""CRC Encoder used for internal validation."""
  ########### Public methods and properties
  def forward(self, inputs):#inputs [...,k+crc_degree], tf.float32: Tensor containing CRC encoded bits (i.e., last `crc_degree` bits are parity bits).  rank Must >= 2.
    """Verifies CRC of `inputs`. Returns result of CRC validation & removes parity bits from ``inputs``.
    Returns:(x, crc_valid): Tuple: 
      x : [...,k], tf.float32 : info bit seq without CRC parity bits.
      crc_valid : [...,1], tf.bool:result of the CRC per codeword.
    """
    inputs = tc.from_numpy(inputs)
    # Assert that the rank of the inputs tensor is at least 2
    assert len(inputs.shape) >= 2,  "Input tensor must have at least rank 2."
    # last dim must be at least crc_bits long
    assert inputs.shape[-1] >= self._encoder.crc_length, f"Last dimension of inputs must be at least {self._encoder.crc_length}."
    # re-encode info bits of x and verify that CRC bits are correct
    x_info = inputs[..., :-self._encoder.crc_length]
    x_parity = self._encoder(inputs)[..., -self._encoder.crc_length:]
    # return if x fulfils the CRC
    crc_check = tc.sum(x_parity, dim=-1, keepdim=True)
    crc_check = tc.where(crc_check>0,False,True)  # True if all parity bits are zero
    # crc_check (3,16,1)
    # x_info (3,16,64)
    return x_info.numpy(), crc_check.numpy()