from pyrallis import field
from dataclasses import dataclass
from typing import List
import torch as tc
@dataclass
class PolarConfig:
  '''
  python x_run_sn_polar/test_all.py --spec True --verbose True
  python x_run_sn_polar/main.py --k 32 --n 64 \
  --algos ['tre_l_F4','tre_F4','rmld_F4'] --list_size 8\
  --bs 1000 --mc_iter 1
  '''
  # code parameters
  k:int = 32  # number of information bits per codeword
  n:int = 64 # desired codeword length 
  algos:List[str] = field(default=
    ['tre_F4','rmld_F4','scl'], is_mutable=True)
  kern:str = 'F2' #only used in test
  verbose:bool = False
  bs:int = 3  #10000
  snr_end:float = 5
  mc_iter:int = 10
  list_size:int = 8  # scl list_size
  mode:str = "max" #* llr
  spec:bool = False  #*apply special case 1,2
device = 'cuda' if tc.cuda.is_available() else 'cpu'
device= 'cpu'