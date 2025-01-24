import torch as tc
import numpy as np
import time
def hard_decisions(llr):
  hard = tc.where(llr>0,1.,0.)
  return hard
def count_block_errors(b, b_hat): #b,b_hat: tc.fp32: tensor with ones/zeros.
  """Counts the number of block errors between two binary tensors.
  A block error happens if at least 1 element of `b`,`b_hat` differ in one block. 
  The BLER is evaluated over last dim of input, i.e., all elements of last dim are a block.
  """
  errors = tc.not_equal(b,b_hat).to(tc.int64)
  block_errs = tc.any(errors,dim=-1)
  return tc.sum(block_errs)
def count_errors(b, b_hat): #b,b_hat: tc.fp32: tensor with ones/zeros.
  """num bit errors between two binary tensors."""
  errors = tc.not_equal(b,b_hat).to(tc.int64)
  return tc.sum(errors)
def sim_ber(mc_fun,#fn yields tx bits `b` & rx estimate `b_hat` for given `bs`&`ebno_db`.
        #If ``soft_estimates`` is True, b_hat is interpreted as logit.
    ebno_dbs, #tc.float32: tensor containing SNR points to be evaluated.
    batch_size,max_mc_iter, #tc.int32: Max. number of Monte-Carlo iterations per SNR point
    soft_estimates=False,#If True,`b_hat`is interpreted as logit & hard-decision is applied.
    target_bit_errs=None,#Target num bit errs per SNR point til simu continues to next SNR point.
    target_block_errs=None,#Target num of block errs per SNR point until simu continues
    early_stop=True,#If True, simu stops after first error-free  
      # SNR point (i.e., no error occurred after `max_mc_iter` Monte-Carlo iterations).
    verbose=True,  
    dtype=tc.complex64,#tc.complex64#Datatype of the model / function to be used (``mc_fun``).
    device='cpu'
  ):
  # utility function to print progress
  def _print_progress( is_final, # If True, the progress is printed into a new line.
    rt,   #float: runtime of the current SNR point in seconds.
    idx_snr,  #int: Idx of current SNR point.
    idx_it,   #int: Current iter idx.
    header_text=None#[str,]:Elements will be printed instead of current progress, 
    #iff not None. Can be used to generate table header
  ):
    """Print summary of current simulation progress."""
    end_str = "\n" if is_final else "\r"# set carriage return if not final step
    # prepare to print table header
    if header_text is not None:  row_text = header_text;  end_str = "\n"
    else: # calculate intermediate ber / bler
      ber_np = (bit_errors[idx_snr].to(tc.float64)
                  / nb_bits[idx_snr].to(tc.float64)).cpu().numpy()
      ber_np = np.nan_to_num(ber_np) # avoid nan for first point
      bler_np = (block_errors[idx_snr].to(tc.float64)
                  / nb_blocks[idx_snr].to(tc.float64)).cpu().numpy()
      bler_np = np.nan_to_num(bler_np) # avoid nan for first point
      # load statuslevel
      # print current iter if simulation is still running
      if status[idx_snr]==0: status_txt = f"iter: {idx_it:.0f}/{max_mc_iter:.0f}"
      else:   status_txt = status_levels[int(status[idx_snr])]
      # generate list with all elements to be printed
      row_text = [str(np.round(ebno_dbs[idx_snr].cpu().numpy(), 3)),
                  f"{ber_np:.4e}",  f"{bler_np:.4e}",
                  np.round(bit_errors[idx_snr], 0),   np.round(nb_bits[idx_snr], 0),
                  np.round(block_errors[idx_snr], 0), np.round(nb_blocks[idx_snr], 0),
                  np.round(rt, 1),  status_txt]
    print("{: >9} |{: >11} |{: >11} |{: >12} |{: >12} |{: >13} |{: >12} |{: >12} |{: >10}".format(*row_text), end=end_str)
   # init table headers
  header_text = ["EbNo [dB]", "BER", "BLER", "bit errors",
       "num bits", "block errors", "num blocks", "runtime [s]", "status"]# replace status by text
                    # status=0       # status=1; spacing for impr. layout
  status_levels = ["not simulated", "reached max iter       ", 
    # status=2                  # status=3                   # status=4
    "no errors - early stop", "reached target bit errors", "reached target block errors"]
  
  ebno_dbs= tc.from_numpy(ebno_dbs).to(tc.float32)  #todo tc.complex.real_dtype
  num_points = ebno_dbs.shape[0] #10
  bit_errors = tc.zeros([num_points],dtype=tc.int64,device=device)
  block_errors = tc.zeros([num_points], dtype=tc.int64,device=device)
  nb_bits = tc.zeros([num_points], dtype=tc.int64,device=device)
  nb_blocks = tc.zeros([num_points], dtype=tc.int64,device=device)
  # track status of simulation (early termination etc.)
  status = tc.zeros(num_points)   # measure runtime per SNR point
  runtime = tc.zeros(num_points)  # ensure num_target_errors is a tensor
  for i in range(num_points):
    runtime[i] = time.perf_counter() # save start time
    iter_count = -1 # for print in verbose mode
    for ii in range(max_mc_iter):
      iter_count += 1
      b,b_hat  = mc_fun(batch_size=batch_size, ebno_db=ebno_dbs[i])# (100,4)
      # (bs,k)
      
      # assume 1st & 2nd return value is b & b_hat, other returns are ignored
      if soft_estimates:  b_hat = hard_decisions(b_hat)
      # count errors
      bit_e = count_errors(b, b_hat)
      block_e = count_block_errors(b, b_hat)
      
      bit_n = tc.numel(b)  #bs*k
      block_n = int(tc.numel(b)/b.shape[-1])
      # update variables
      bit_errors[i] += bit_e; block_errors[i] += block_e
      nb_bits[i] += bit_n;    nb_blocks[i] += block_n
      if verbose:
        if i==0 and iter_count==0:# print summary header during first iteration
          _print_progress(is_final=True, rt=0,idx_snr=0,idx_it=0, header_text=header_text)
          print('-' * 135)# print seperator after headline
        # evaluate current runtime
        rt = time.perf_counter() - runtime[i]
        # print current progress
        _print_progress(is_final=False, idx_snr=i, idx_it=ii, rt=rt)
      
      if target_bit_errs is not None:# bit-error based stopping cond.
        if bit_errors[i]>= target_bit_errs:
          status[i] = 3 # change internal status for summary
          # stop runtime timer
          runtime[i] = time.perf_counter() - runtime[i]
          break # enough errors for SNR point have been simulated
      if target_block_errs is not None:# block-error based stopping cond.
        if block_errors[i]>=target_block_errs:
          # stop runtime timer
          runtime[i] = time.perf_counter() - runtime[i]
          status[i] = 4 # change internal status for summary
          break # enough errors for SNR point have been simulated
      # max iter have been reached -> continue with next SNR point
      if iter_count==max_mc_iter-1: # all iterations are done
          # stop runtime timer
          runtime[i] = time.perf_counter() - runtime[i]
          status[i] = 1 # change internal status for summary
    
    # print results again AFTER last iteration / early stop (new status)
    if verbose:_print_progress(is_final=True, idx_snr=i, idx_it=iter_count, rt=runtime[i])
    # early stop if no error occurred
    if early_stop: # only if early stop is active
      if block_errors[i]==0:
        status[i] = 2 # change internal status for summary
        if verbose:
          print(f"\nSimu stopped as no error occurred @ EbNo = {ebno_dbs[i].numpy():.1f} dB.\n")
        break  
  ber = bit_errors / nb_bits# calculate BER / BLER
  bler = block_errors / nb_blocks

  # replace nans (from early stop)
  ber = tc.where(tc.isnan(ber), tc.zeros_like(ber), ber)
  bler = tc.where(tc.isnan(bler), tc.zeros_like(bler), bler)
  return ber, bler