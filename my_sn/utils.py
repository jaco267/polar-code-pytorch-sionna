import torch as tc
def complex_normal(shape, var=1.0,device='cpu'):
  r"""Generates a tensor of complex normal random variables.
  Input-----
    shape : tc.shape, or list: The desired shape.  (bs,n/n_sym)
    var:float: total variance.,i.e.,each complex dim has variance ``var/2``.
  Output------
    Tensor of complex normal random variables.
  """
  # Half the variance for each dimension
  var_dim = tc.tensor(var/2,dtype=tc.float32)
  stddev = tc.sqrt(var_dim)
  # Generate complex Gaussian noise with the right variance
  xr = tc.normal(mean=0,std=stddev,size=shape, dtype=tc.float32,device=device)
  xi = tc.normal(mean=0,std=stddev,size=shape, dtype=tc.float32,device=device)
  x = tc.complex(xr, xi)
  return x

def expand_to_rank(tensor, target_rank, axis=-1):
  """Inserts as many axes to a tensor as needed to achieve a desired rank.
  inserts extra dims to `tensor` starting at `axis`, 
  so that rank of resulting tensor has rank `target_rank`. 
  Args:
    tensor : A tensor.
    target_rank (int) : rank of output tensor.
      If `target_rank` is smaller than rank of `tensor`, do nothing.
    axis (int) : dim index at which to expand shape of `tensor`. 
    Given a ``tensor`` of `D` dims,`axis` must be in range`[-(D+1), D]`(inclusive).
  Returns:
    tensor with same data as `tensor`, with `target_rank`- rank(`tensor`) 
    extra dims inserted at index specified by ``axis``.
    If ``target_rank`` <= rank(``tensor``), ``tensor`` is returned.
  """
  num_dims = max(target_rank - len(tensor.shape), 0)
  output = insert_dims(tensor, num_dims, axis)
  return output
def insert_dims(tensor, num_dims, axis=-1):
  """Adds multiple length-one dims to a tensor.
  It inserts ``num_dims`` dimensions of length one starting from the
  dimension ``axis`` of a ``tensor``.
  Args:
    tensor : A tensor.
    num_dims (int) : num dims to add.
    axis : dim index at which to expand shape of `tensor`. 
    Given a ``tensor`` of `D` dims,`axis` must be in range`[-(D+1), D]`(inclusive).
  Returns:
    A tensor with the same data as ``tensor``, with ``num_dims`` additional
    dimensions inserted at the index specified by ``axis``.
  """
  assert num_dims >= 0, "`num_dims` must be nonnegative."
  rank = len(tensor.shape)
  assert -(rank+1) <= axis<= rank, "`axis` is out of range `[-(D+1), D]`)"

  axis = axis if axis>=0 else rank+axis+1
  shape = tensor.shape
  new_shape = list(shape[:axis])+[1]*num_dims+list(shape[axis:])
  output = tensor.reshape(new_shape)
  return output
