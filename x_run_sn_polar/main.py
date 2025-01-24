#python examples/b_polar_5G/main_sc.py --option ['no_code','rm_sc','rm_scl']
import sys
from os.path import dirname
proj_path = dirname(dirname(dirname(__file__)))
sys.path.append(proj_path)
proj_path = dirname(dirname(__file__))
sys.path.append(proj_path)
print(proj_path)
import torch as tc
'''llr > 0  ->1  
llr < 0  ->0'''
import numpy as np
import matplotlib.pyplot as plt
from my_sn.plotting import PlotBER
from polar.enc import PolarEncoder as PolarEnc2
from polar.polar_scl import SCL_Dec
from polar.polar_sc import SC_Dec
import pyrallis
import math
from config import PolarConfig,device
from z_sys_model.awgn_model import System_AWGN_model

import random
seed = 42  
def set_seed(seed):
  np.random.seed(seed)
  random.seed(seed)
  tc.manual_seed(seed)
set_seed(seed)
from d_kernels import *#F2,F4,G4,F8,G8,F16,G16,F32,G32,G162
from polar.froze import get_Kern_frozen_bits,get_Kern_frozen_bits2
def gen_code(c:PolarConfig,Gn,name,mode='vit'):
  a = math.log(c.n,2);  assert a.is_integer()#7
  G_,G_weights,frozen_pos = get_Kern_frozen_bits(c.n,c.n-c.k,Gn)
  enc = PolarEnc2(frozen_pos, c.n,G_,device=device)  # RM codes with SCL decoding
  # for i in frozen_pos: print(f"1 {i}")
  if mode=="sc":  dec = SC_Dec(frozen_pos, c.n,device=device) 
  elif mode=="scl": dec = SCL_Dec(frozen_pos, c.n,c.list_size,device=device)
  else: raise Exception('error...') 
  rm_scl_awgn = System_AWGN_model(c.n,c.k,enc,dec,device=device)
  return [rm_scl_awgn, name]
@pyrallis.wrap()    
def main(c: PolarConfig):
  k = c.k; n = c.n; print(c.algos,type(c.algos))  
  ebno_db = np.arange(0, c.snr_end, 0.5) # sim SNR range
  codes_under_test = []
  if 1: # 'rm_scl' in c.algos:  # rm scl
    code = gen_code(c,F2,"SC",mode="sc")
    codes_under_test.append(code)
  if 'scl' in c.algos:  # rm scl
    code = gen_code(c,F2,f"SCL-{c.list_size}",mode="scl")
    codes_under_test.append(code)
  ber_plot128 = PlotBER(f"Performance of Short Len Codes (k={k}, n={n})")
  # run ber simulations for each code we have added to the list
  for code in codes_under_test:
    print("\nRunning: " + code[-1])
    set_seed(seed)
    ber_plot128.simulate(code[0],ebno_dbs=ebno_db,batch_size=c.bs, target_block_errs=1000, 
      legend=code[-1],soft_estimates=False,max_mc_iter=c.mc_iter,add_bler=True,device=device) 
  fig, ax = plt.subplots(figsize=(16,12))# generate new figure
  plt.xticks(fontsize=18); plt.yticks(fontsize=18)
  plt.title(f"SC vs scl (k={k},n={n})", fontsize=25)
  plt.grid(which="both")
  plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25);plt.ylabel(r"BLER", fontsize=25)
  for i in range(len(ber_plot128.legend)): # 1
    if "BLER" in ber_plot128.legend[i]:
      if "SC" in ber_plot128.legend[i]: linestyle='--'
      else: linestyle = '-'
      plt.semilogy(ebno_db, ber_plot128.ber[i],c='C%d'%(i), label=ber_plot128.legend[i],linewidth=2,linestyle=linestyle)
    else: pass
      # plt.semilogy(ebno_db,ber_plot128.ber[i],c='C%d'%(i),label= ber_plot128.legend[i],linestyle = "--",linewidth=2)

  plt.legend(fontsize=20)
  plt.xlim([0, 4.5])
  plt.savefig(f'./x_run_sn_polar/plots/sc_{c.mc_iter=}_{c.bs=}.png')
  plt.show()
if __name__ == '__main__':
  main()
  