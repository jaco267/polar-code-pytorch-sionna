import matplotlib.pyplot as plt
from .sim import sim_ber
def plot_ber(plot_self,  ylabel="BER"
  ): # Plot error-rates # tile snr_db if not list, but ber is list
  snr_db = plot_self.snr# np fp32: simulated SNR points.
  ber = plot_self.ber   # np fp32: BER/BLER per SNR point.
  legend = plot_self.legend   
  title = plot_self.title
  fig, ax = plt.subplots(figsize=(16,10))
  plt.xticks(fontsize=18); plt.yticks(fontsize=18);
  plt.title(title, fontsize=25)
  for idx, b in enumerate(ber):  
     
     plt.semilogy(snr_db[idx], b, linewidth=2)# return figure handle
  plt.grid(which="both"); plt.xlabel(r"$E_b/N_0$ (dB)", fontsize=25)
  plt.ylabel(ylabel, fontsize=25);  plt.legend(legend, fontsize=20)
  return fig, ax
class PlotBER():#"""Provides a plotting object to simulate and store BER/BLER curves"""
  def __init__(self, title="Bit/Block Error Rate"): #title of the figure
    self.title = title
    self.ber = [];self.snr = [];self.legend = [];#List of all stored BER/SNR/legend entries curves.
  def simulate(self,mc_fun,#fn yields tx bits `b` & rx estimate `b_hat` for given `bs`&`ebno_db`. 
    ebno_dbs,#np.fp SNR points to be evaluated
    batch_size, legend="", 
    add_ber=True,
    add_bler=False,
    max_mc_iter=1,#Max num of Monte-Carlo iterations per SNR point.
    soft_estimates=False,#If True,`b_hat` is logit & additional hard-decision is applied internally.
    target_bit_errs=None,target_block_errs=None,#Target bit(block) errs per SNR point til simu stops
    verbose=True,
    device='cpu'
  ):
    """Simulate BER/BLER curves for given Keras model and saves results.
    Output------  (ber, bler):Tuple:
    ber: float The simulated bit-error rate.
    """
    ber, bler = sim_ber(mc_fun, ebno_dbs,batch_size,
        soft_estimates=soft_estimates,max_mc_iter=max_mc_iter,
        target_bit_errs=target_bit_errs,
        target_block_errs=target_block_errs,verbose=verbose,device=device)
    if add_ber:
      self.ber += [ber]; 
      self.snr +=  [ebno_dbs]; self.legend += [legend];
    if add_bler:
      self.ber += [bler]
      self.snr +=  [ebno_dbs]; self.legend += [legend+" (BLER)"];
      
    return ber, bler
