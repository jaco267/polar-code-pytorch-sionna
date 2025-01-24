from my_sn.trans.channel import awgn
from my_sn.trans import mapping,binary_source,ebno
from torch import nn
from numpy.random import randint
import math
import torch as tc
import numpy as np 
e2max = (2 ** 32) - 1
def generate_gaussian_noise(mean, stddev):
  # Generate two uniformly distributed random numbers between 0 and 1
  u1 = randint(256**4, dtype='<u4', size=1)[0]/e2max
  u2 = randint(256**4, dtype='<u4', size=1)[0]/e2max
  # Apply Box-Muller transform
  z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
  return z0 * stddev + mean
class System_AWGN_model(nn.Module): # Inherits from Keras Model
  def __init__(self,n,k,encoder,decoder,
    cw_estimates=False,#If true, codewords instead of info estimates are returned.
    device='cpu'
  ):
    super().__init__() # Must call the Keras model initializer
    self.cw_estimates=cw_estimates # if true codewords instead of info bits are returned
    self.n_bits_per_sym = 2
    self.n = n # codeword len per transmitted codeword
    self.k = k # number of info bits per codeword
    self.coderate = self.k/self.n
    self.constell = mapping.QamConstell(self.n_bits_per_sym,device=device)
    self.mapper = mapping.Mapper(constell=self.constell,device=device)
    self.demapper = mapping.Demapper(constell=self.constell)
    self.binary_src = binary_source.BinarySource(device=device)
    self.awgn_channel = awgn.AWGN(device=device)
    self.encoder = encoder; self.decoder = decoder
  def forward(self, batch_size, ebno_db):
    no = ebno.ebnodb2no(ebno_db,self.n_bits_per_sym,self.coderate)
    bits = self.binary_src([batch_size, self.k])
    codewords = self.encoder(bits) 
    x = self.mapper(codewords)
    
    y = self.awgn_channel([x, no])
    llr = self.demapper([y,no])
    bits_hat = self.decoder(llr)
    if self.cw_estimates:
        return codewords, bits_hat
    return bits, bits_hat
  # def forward(self, batch_size, ebno_db): #* winproc mode  
  #   no = ebno.ebnodb2no(ebno_db,self.n_bits_per_sym,self.coderate)
  #   bits = self.binary_src([batch_size, self.k])
  #   codewords = self.encoder(bits) 
  #   x = self.mapper(codewords)

  #   nnoise = tc.zeros(codewords.shape)
  #   for bs in range(nnoise.shape[0]):
  #     for nnn in range(nnoise.shape[1]):
  #        nnoise[bs,nnn]=generate_gaussian_noise(0,tc.sqrt(no).numpy())
  #   pTransmitted = (2*codewords-1) #*mapper
  #   pTransmitted += nnoise  
  #   pTransmitted = -2*pTransmitted/no   
  #   llr = -pTransmitted
  #   # y = self.awgn_channel([x, no])
  #   # llr = self.demapper([y,no])
  #   bits_hat = self.decoder(llr)
  #   if self.cw_estimates:
  #       return codewords, bits_hat
  #   # breakpoint()
  #   return bits, bits_hat