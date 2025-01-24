"""Utility functions and layers for the Polar code package."""
import numpy as np
from scipy.special import comb
from importlib_resources import files, as_file
from . import codes
def generate_5g_ranking(k, n, sort=True,strict=True):
  """Returns info and frozen bit positions of the 5G Polar code as defined 
  in Tab. 5.3.1.2-1 in [3GPPTS38212]_ for given values of `k` and `n`.
  Input-----
    k: int :  info bit num per codeword.
    n: int : desired codeword len. Must be a power of two.
    sort: bool=True. Indicates if the returned indices are sorted.
  Output------[frozen_pos, info_pos]: List:
    frozen_pos: ints array `[n-k]` of frozen position indices.
    info_pos:   int array  `[k]` of information position indices.
  RaisesError  If ((k or n) > 1024) || (n < 32) || (coderate >1.0) || n !=  power of 2.
  in the spec  
  channel reliability (0 bad,1024 good); idx in codeword 
  0;0
  1;1
  2;2
  3;4
  4;8
  5;16
  6;32
  7;3
  8;5
  9;64
  10;9
  11;6
  12;17
  13;10
  14;18
  15;128
  16;12
  17;33
  18;65
  19;20
  20;256
  21;34
  22;24
  23;36
  24;7
  """
  if strict:
    assert k<1025,"k cant > 1024."; assert n<1025,"n cant > 1024.";assert n>31,"n cant < 32."; 
    assert n>=k, "Invalid coderate (>1)."; assert np.log2(n)==int(np.log2(n)), "n must be a power of 2."
  source = files(codes).joinpath("polar_5G.csv")# load channel ranking from csv file  in folder "codes"
  with as_file(source) as codes.csv:
      ch_order = np.genfromtxt(codes.csv, delimiter=";")
  ch_order = ch_order.astype(int)
  # find n smallest values of channel order (2nd row)
  ind = np.argsort(ch_order[:,1]) #[0,1,2,7,3,8,11,24,4,10..])
  ch_order_sort = ch_order[ind,:]
  # only consider the first n channels (ex.n=8,k=4)
  ch_order_sort_n = ch_order_sort[0:n,:]  #idx 0 1 2 3 4 5 6 7
                            #chan reliability [0,1,2,7,3,8,11,24]
                            #               worst             best
  # and sort again according to reliability
  ind_n = np.argsort(ch_order_sort_n[:,0]) #[0, 1, 2, 4, 3, 5, 6, 7]
  ch_order_n = ch_order_sort_n[ind_n,:]
  # and calculate frozen/info positions for given n, k
  # assume that pre_frozen_pos are already frozen (rate-matching)
  frozen_pos = np.zeros(n-k)
  info_pos = np.zeros(k)
  #the n-k smallest positions of ch_order denote frozen pos.
  for i in range(n-k):   frozen_pos[i] = ch_order_n[i,1] # 2. row yields index to freeze
  for i in range(n-k, n):info_pos[i-(n-k)] = ch_order_n[i,1] # 2. row yields index to freeze
  # sort to have channels in ascending order
  if sort:info_pos = np.sort(info_pos); frozen_pos = np.sort(frozen_pos)
  return [frozen_pos.astype(int), info_pos.astype(int)] #froz(0,1,2,4)  info [3,5,6,7]

def generate_rm_code(
  r, #int:  The order of the RM code.
  m  #int: `log2` of the desired codeword length.
):
  """Generate frozen positions of the (r, m) Reed Muller (RM) code.
  Output------ [frozen_pos, info_pos, n, k, d_min]:
  frozen_pos: npints shape `[n-k]` : frozen position indices.
  info_pos: np ints shape `[k]` : the info position indices.
  n: int  Resulting codeword length
  k: int Number of info bits
  d_min: int Minimum distance of the code.
  """
  assert r<=m, "order r cannot be larger than m."
  n = 2**m;   d_min = 2**(m-r)   # calc k to verify results
  k = 0
  for i in range(r+1):  k += int(comb(m,i))
  # select positions to freeze # freeze all rows that have weight < m-r
  w = np.zeros(n)
  for i in range(n):
    x_bin = np.binary_repr(i)
    for x_i in x_bin: w[i] += int(x_i)
  frozen_vec = w < m-r
  info_vec = np.invert(frozen_vec)
  k_res = np.sum(info_vec)
  frozen_pos = np.arange(n)[frozen_vec]
  info_pos = np.arange(n)[info_vec]       #len=n/2
  # verify results
  assert k_res==k, "Error: resulting k is inconsistent."   
  return frozen_pos, info_pos, n, k, d_min