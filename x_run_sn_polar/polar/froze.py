import numpy as np
import math  
import torch as tc
def get_Kern_frozen_bits(n,f_num,kern):
  '''generate G_arikan matrix and rm froze bits(rows with least weights)'''
  base = kern.shape[0]
  _nb_stages = int(math.log(n,base))
  assert base**_nb_stages == n, f"{n=}, is not power of {base=}"
  m = tc.clone(kern)
  for _ in range(_nb_stages-1):
    m = tc.kron(kern,m)
  G = m
  G_weights = tc.sum(G,dim=1)
  frozen_pos = tc.sort(tc.argsort(G_weights)[:f_num])[0]#.numpy()
  #**just select least weights rows as frozen_pos
  return G,G_weights,frozen_pos
def get_Kern_frozen_bits2(n,f_num,kern):
  
  '''generate G_arikan matrix and rm froze bits(rows with least weights)'''
  base = kern.shape[0]
  _nb_stages = int(math.log(n,base))
  assert base**_nb_stages == n, f"{n=}, is not power of {base=}"
  m = np.copy(kern)
  for _ in range(_nb_stages-1):
    m = np.kron(kern,m)
  G = m
  G_weights = np.sum(G,axis=1)
  frozen_pos = np.sort(np.argsort(G_weights)[:f_num])#.numpy()s
  #**just select least weights rows as frozen_pos
  return G,G_weights,frozen_pos


