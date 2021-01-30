"""Collections of util functions and classes."""

import numpy as np
import tensorflow as tf
from collections import defaultdict
from typing import List
from functools import wraps


def inplace(args: str):
  """Returns a decorator hinting in-place operation on the arguments of
  the decorated function.

  The argument `args` represents a description of the in-place arguments.
  """

  def decorator(func):

    # the decorated function is the same as the original function
    @wraps(func)
    def decorated(*args, **kwargs):
      return func(*args, **kwargs)

    # ......except for an additional line in the docstring.
    origin_doc = decorated.__doc__
    if origin_doc is None:
      origin_doc = ''
    new_doc = f'In-place function on argument(s) {args}! ' + origin_doc
    decorated.__doc__ = new_doc

    return decorated

  return decorator


def inner(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  return tf.reduce_sum(x * y, axis=-1)


def outer(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  return tf.expand_dims(x, axis=-1) * tf.expand_dims(y, axis=-2)


def random(shape: List[int], seed: int) -> tf.Tensor:
  return tf.random.uniform(shape=shape, minval=0., maxval=1., seed=seed)


def expect(x: tf.Tensor):
  return tf.reduce_mean(x, axis=0)


def create_variable(name: str,
                    shape: List[int],
                    initializer: tf.initializers.Initializer,
                    dtype: str = 'float32',
                    **variable_kwargs):
  init_value = initializer(shape, dtype)
  return tf.Variable(init_value, name=name, **variable_kwargs)


class History:

  def __init__(self):
    self.logs = defaultdict(dict)

  def log(self, step: int, key: str, value: object):
    try:  # maybe a tf.Tensor
      value = value.numpy()
    except AttributeError:
      pass
    self.logs[step][key] = value

  def show(self, step: int, keys: List[str] = None):
    """Returns the string to show."""
    if keys is None:
      keys = list(self.logs[step])

    aspects = []
    for k in keys:
      v = self.logs[step].get(k, None)
      if isinstance(v, (float, np.floating)):
        v = f'{v:.5f}'
      elif isinstance(v, str):
        pass
      else:
        raise ValueError(f'Type {type(v)} is temporally not supported.')
      aspects.append(f'{k}: {v}')

    show_str = ' - '.join([f'step: {step}'] + aspects)
    return show_str


def get_sparsity_constraint(sparsity: float, seed: int):

  if not sparsity:
    return None

  mask = None

  def sparsity_constraint(kernel: tf.Tensor) -> tf.Tensor:
    nonlocal mask
    if mask is None:
      rand = random(shape=kernel.shape, seed=seed)
      mask = tf.where(rand > sparsity, 1, 0)
      mask = tf.cast(mask, kernel.dtype)
    return kernel * mask

  return sparsity_constraint
