import torch as tc
def int_mod_2(x):
  r"""Efficient implementation of modulo 2 operation for integer inputs.
  This function assumes integer inputs or implicitly casts to int.
  Remark: the function `tf.math.mod(x, 2)` is placed on the CPU and, thus,
  causes unnecessary memory copies.
  Parameters----------
  y = x%2
  x: tf.Tensor Tensor to which the modulo 2 operation is applied.
  """
  x_int32 = x.to(dtype=tc.int32)
  y_int32 = tc.bitwise_and(x_int32, tc.tensor(1, dtype=tc.int32))  
  return y_int32.to(dtype=x.dtype)
