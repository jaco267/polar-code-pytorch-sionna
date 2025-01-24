from my_sn.trans.channel import awgn
from my_sn.trans.channel.discrete_channel import BinaryErasureChannel
from my_sn.trans import mapping,binary_source,ebno
from config import PolarConfig
from torch import nn
class System_BEC_model(nn.Module): # Inherits from Keras Model
  def __init__(self,c:PolarConfig,encoder,decoder,
    cw_estimates=False,#If true, codewords instead of info estimates are returned.
    device='cpu'
  ):
    super().__init__() # Must call the Keras model initializer
    self.cw_estimates=cw_estimates # if true codewords instead of info bits are returned
    self.c = c; 
    self.n = c.n # codeword len per transmitted codeword
    self.k = c.k # number of info bits per codeword
    self.coderate = self.k/self.n
    self.binary_src = binary_source.BinarySource(device=device)
    self.channel = BinaryErasureChannel(return_llrs=True)
    self.encoder = encoder; self.decoder = decoder
  def forward(self, batch_size, ebno_db):#ebno_db:pe: p erasure
    bits = self.binary_src([batch_size, self.k])
    codewords = self.encoder(bits) 
    llr = self.channel([codewords, ebno_db])
    bits_hat = self.decoder(llr)
    if self.cw_estimates:
        return codewords, bits_hat
    return bits, bits_hat