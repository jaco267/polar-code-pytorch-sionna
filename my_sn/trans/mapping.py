"""Layers for (de)mapping, constellation class, and utility functions"""
import numpy as np
import matplotlib.pyplot as plt
from ..utils import expand_to_rank
import torch as tc
from torch import nn
def pam_gray(b):#[n], np Tensor with with binary entries.
  r"""Maps a vec of bits to PAM constell points with Gray labeling.
  maps bin vector to Gray-labelled PAM constell points.
  used to generated QAM constell. constell isn't normalized.
  Output--:int: PAM constell point taking values in {+-1,3,..+-(2^n-1)}
  """ 
  if len(b)>1:return (1-2*b[0])*(2**len(b[1:]) - pam_gray(b[1:]))
  return 1-2*b[0]  #0->1, 1->-1
def qam(n_bits_per_sym, #(constell point):int:Must be multiple of 2, e.g 2, 4, 6 etc.
        normalize=True #If`True`,constell is normalized to have unit power.
  ):
  r"""Gen a QAM constell(complex-val vec), each element is constell point of
  an M-ary QAM constell. bit label of `n`th point is given by 
  len(n_bits_per_sym) bin represenation of `n`.
  Output ------len(2^n_bits_per_sym), np.complex64, The QAM constell.
  Note----
  bit label of nth constell point is given by bin represent of its 
  position within array,can be obtained by`np.binary_repr(n, n_bits_per_sym)`.
  normalization factor of a QAM constell is given in closed-form as:
  sqrt( 1/(2^(n-2))  sum (2i-1)^2)
  n= n_bits_per_sym/2  is num of bits per dim.
  """
  # ex. n_bits_per_sym=2 -> 2^2= 4 Qam
  assert n_bits_per_sym % 2 == 0 and n_bits_per_sym >0 # is even
  c = np.zeros([2**n_bits_per_sym], dtype=np.complex64)#Build constell by iter through all points
  for i in range(0, 2**n_bits_per_sym):
    b = np.array(list(np.binary_repr(i,n_bits_per_sym)), dtype=np.int16)
    # [0 0] -> [0 1] -> [1 0] -> [1 1]
    '''              pam gray
     0 0   0 1       1+1j   1-1j 
                 ->                  
     1 0   1 1      -1+1j  -1-1j 
    '''
    c[i] = pam_gray(b[0::2]) + 1j*pam_gray(b[1::2]) # PAM in each dim
  
  if normalize: # Normalize to unit energy
    n = int(n_bits_per_sym/2)  #2/2=1
    # ex. for n=4 16 qam    -> sum 1^2+3^2 = 1+9=10
    qam_var = 1/(2**(n-2))*np.sum(np.linspace(1,2**n-1, 2**(n-1))**2)
    # 2.0
    c /= np.sqrt(qam_var)   #c/=sqrt(2)
  return c  #len(c)==2^(n_bits_per_sym)
class QamConstell(nn.Module):
  r"""Constell that can be used by a (de)mapper.
  defines a constell, i.e,complex vec of constell points. 
  binary idx of element of this vec corresponds to 
  bit label of constell point.
  Output------ 2^{n_bits_per_symbol}, `dtype`The constellation.
  """
  def __init__(self,
    n_bits_per_symbol,#int:num_bits per constell symbol,e.g,4->QAM16.
    normalize=True,#If `True`, constell is normalized to have unit power.
    dtype=tc.complex64, # dtype of the constell.
    device= 'cpu'
  ):
    super().__init__()
    self.dtype = dtype
    self.normalize = normalize#check if constell is normalized or not
    # allow float inputs that represent int
    assert isinstance(n_bits_per_symbol, (float,int)),"n_bits_per_sym must be integer"
    assert (n_bits_per_symbol%1==0),"n_bits_per_symbol must be integer"
    n_bits_per_symbol = int(n_bits_per_symbol)
    assert n_bits_per_symbol%2 == 0 and n_bits_per_symbol>0,\
        "n_bits_per_symbol must be a multiple of 2"
    self.n_bits_per_sym = int(n_bits_per_symbol)#num of bits per constell symbol
    points = qam(self.n_bits_per_sym, normalize=self.normalize)
    self.device = device
    points = tc.from_numpy(points).to(dtype=self.dtype,device=device)
    # build 
    points=tc.stack([tc.real(points),tc.imag(points)], dim=0)
    self._points = points.to(dtype=tc.float32)
  def forward(self, inputs=None):
    # assert int(inputs.numpy())==100
    x = self._points
    x = tc.complex(x[0], x[1])
    if self.normalize:
      energy = tc.mean(tc.abs(x)**2)
      energy_sqrt = tc.sqrt(energy).to(dtype=self.dtype)
      x = x / energy_sqrt
    return x
  @property
  def points(self): return self(None)#(possibly) centered and normalized constell points
  def show(self, labels=True, figsize=(7,7)):
    """Generate a scatter-plot of the constellation.
    labels : bool: bit labels be drawn next to each constellpoint. 
    """
    maxval = np.max(np.abs(self.points))*1.05
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plt.xlim(-maxval, maxval)
    plt.ylim(-maxval, maxval)
    plt.scatter(np.real(self.points), np.imag(self.points))
    ax.set_aspect("equal", adjustable="box")
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True, which="both", axis="both")
    plt.title("Constellation Plot")
    if labels is True:
        for j, p in enumerate(self.points.numpy()):
            plt.annotate(
                np.binary_repr(j, self.n_bits_per_sym),
                (np.real(p), np.imag(p))
            )
    return fig
class Mapper(nn.Module):
  r"""Maps binary tensors to points of a constell.
  maps bin val tensor to tensor of points from provided constell.
  Input----: [..., n], tc.fp/int: Tensor with with binary entries.
  Output------
  : [...,n/Constell.n_bits_per_symbol], tc.complex
    mapped constellation symbols.
  : [...,n/Constell.n_bits_per_symbol], tc.int32
    symbol indices corresponding to constell symbols.
      Only returned if ``return_indices`` is set to True.
  Note----
  last input dim must be an int multiple of num of bits per constell symbol.
  """
  def __init__(self,
    constell=None, #class:`~sionna.mapping.Constellation`
    dtype=tc.complex64,# output dtype. 
    device = 'cpu'
  ):
    super().__init__()
    # Create constellation object
    assert constell.dtype==dtype, "Constellation has wrong dtype."
    self.constell = constell # Constellation used by the Mapper
    #n_bits_per_symbol(constellation) : int   e.g., 4 for QAM16.
    self.device = self.constell.device
    self._binary_base = 2**tc.tensor(range(self.constell.n_bits_per_sym-1,-1,-1),device=self.device)
  def forward(self, inputs):  #(bs,n=8)
    # Reshape inputs to the desired format
    # breakpoint()
    new_shape = [-1] + list(inputs.shape[1:-1]) + \
       [int(inputs.shape[-1] / self.constell.n_bits_per_sym),
        self.constell.n_bits_per_sym]
    
    # breakpoint()
    inputs_reshaped = inputs.reshape(new_shape).to(tc.int32)
    # Convert the last dimension to an integer
    int_rep = tc.sum(inputs_reshaped*self._binary_base, dim=-1)
    # Map integers to constellation symbols
    x = self.constell.points[int_rep]
    return x

class SymbolLogits2LLRs(nn.Module):
  r"""Computes LLRs or hard-decisions on bits
  from a tensor of logits (i.e., unnormalized log-prob) on constell points.
  Output---:[...,n, n_bits_per_sym], tc.fp32: LLRs for every bit.
  Note---- LLR for ith bit is computed according 
 math::                        c in C_i,1
 LLR(i)=ln(Pr(bi=1|z))=ln(sum exp(z_c)) 
          (Pr(bi=0|z))   (sum exp(z_c))
                               c in C_i,0
  where `C_i,1`,`C_i,0` are sets of `2^K` constell points for which
  ith bit is equal to 1 and 0, respectively. 
  `z=[z_c_0,...z_c_2^{k-1}` is vec of logits on constell points, 
  """
  def __init__(self,
    n_bits_per_sym#int n bits per constell symbol, e.g., 4 for QAM16.
  ):
    super().__init__()
    self.n_bits_per_sym = n_bits_per_sym
    n_points = int(2**n_bits_per_sym)  #ex.16Qam -> 4bits_per sym,n_points=16
    # Array composed of binary representations of all symbols indices
    a = np.zeros([n_points, n_bits_per_sym])  #(4,2)
    for i in range(0, n_points):
      '''
      a = [[0. 0.]     0 
           [0. 1.]     1
           [1. 0.]     2
           [1. 1.]]    3
      '''
      a[i,:] = np.array(list(np.binary_repr(i, n_bits_per_sym)), dtype=np.int16)
    # Compute symbol indices for which the bits are 0 or 1
    c0 = np.zeros([int(n_points/2), n_bits_per_sym])
    c1 = np.zeros([int(n_points/2), n_bits_per_sym])
    for i in range(n_bits_per_sym-1,-1,-1):   #  1,0
      '''              c0      c1        00       10
      a = [[0. 0.]    0  0                o 0   2 o
           [0. 1.]    1         1            x                  02/13  y axis
           [1. 0.]       2    2           o 1   3 o
           [1. 1.]]           3 3        01       11            01/23  x axis 
      '''
      c0[:,i] = np.where(a[:,i]==0)[0]
      c1[:,i] = np.where(a[:,i]==1)[0]
    self._c0 = tc.tensor(c0, dtype=tc.int64) # Symbols with ith bit=0
    self._c1 = tc.tensor(c1, dtype=tc.int64) # Symbols with ith bit=1
    self._reduce = tc.logsumexp #reduce fn for LLR computation
  def forward(self, inputs):
    exponents=inputs  # Compute exponents#[bs,n=4,n_points=4],tc.fp:logits of constell points.
    # shape [...,n,num_points/2,n_bits_per_sym]
    exp0 = exponents[...,self._c0] #exponents[:,:,self._c0]#[bs,n=4,2,2]<-(bs,n,4)
    # takes  points where a  is 0
    exp1 = exponents[...,self._c1] #exponents[:,:,self._c1]
    
    # Compute LLRs using the definition log( Pr(b=1)/Pr(b=0) )
    # shape [..., n, n_bits_per_sym]
    llr = self._reduce(exp1, dim=-2) - self._reduce(exp0, dim=-2)#[bs,n=4,2]
    return llr

class Demapper(nn.Module):
  r"""Computes LLRs on bits for tensor of rcv symbols.
  Output--- : [...,n*n_bits_per_sym], tc.fp: LLRs for every bit.
  Note---- LLR for the ith bit is computed according to
  math::                         c in C_i,1
   LLR(i)=ln(Pr(bi=1|y))=ln( exp(-|y-c|^2 /N0) ) 
            (Pr(bi=0|y))   ( exp(-|y-c|^2)/N0) )
                                 c in C_i,0
  where `C_i,1`,`C_i,0` are sets of constell points which ith bit is 
  equal to 1 and 0, respectively. 
  """
  def __init__(self,constell, dtype=tc.complex64):
    super().__init__() # Create constellation object
    assert constell.dtype==dtype,"Constellation has wrong dtype."
    self.constell=constell#class:`~sionna.mapping.Constellation`
    self.device = self.constell.device
    n_bits_per_sym = self.constell.n_bits_per_sym #int:num bits per constell sym,e.g.,4 for QAM16.
    self._logits2llrs = SymbolLogits2LLRs(n_bits_per_sym)
  def forward(self, inputs):
    y, no = inputs#y : [bs,n=4], tc.complex: The received symbols.
    #no: fp32: noise variance estimate # scalar for entire input batch 
    points_shape = [1]*len(y.shape) + list(self.constell.points.shape)#[1,1]+[4,]# Reshape constell points to [1,1,num_points=4]
    points = self.constell.points.reshape(points_shape)#(4,)->(1,1,4)
    # Compute squared distances from y to all points
    # shape [...,n,num_points]            #|(bs,n=4,1) - (1,1,4)|^2
    squared_dist = tc.abs(y.unsqueeze(dim=-1) - points)**2
      #128,4,4     128
    no = expand_to_rank(no,target_rank=len(squared_dist.shape),axis=-1).to(self.device)
    exponents = -squared_dist/no   # Compute exponents
    llr = self._logits2llrs(exponents)
    # Reshape LLRs to [...,n*n_bits_per_sym]
    #todo
    out_shape = list(y.shape[:-1]) + [y.shape[-1] * self.constell.n_bits_per_sym]
    llr_reshaped = llr.reshape(out_shape)
    return llr_reshaped

