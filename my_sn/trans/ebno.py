import torch as tc
def ebnodb2no(ebno_db, n_bits_per_sym, coderate):
  r"""Compute noise variance `No` for given `Eb/No` in dB.
  The fn takes into account num of coded bits per constell symbol, the coderate
  value of `No` is computed according to the following expression
  N_o = (E_b r  M) ^-1
        (n_o  E_s)
  `2^M`:constell size,`M` is avg num of coded bits per constell symbol
  `E_s=1`: avg energy per constell per symbol
  `r`:   coderate
  `E_b`: energy per info bit,
  `n_o`: noise power spectral density.
  `E_s` is for OFDM (otherwise == 1) 
  Input-----
  ebno_db :float: `Eb/No` value in dB.
  num_bits_per_symbol : int: number of bits per symbol.
  coderate : float: The coderate used.
  Output------:float: value of `N_o` in linear scale.
  """
  dtype = tc.float32
  ebno = 10.**(ebno_db/10.)  #E_b/n_o
  energy_per_symbol = 1  #E_s
  no = 1/(ebno*coderate*n_bits_per_sym/energy_per_symbol)
  return no

