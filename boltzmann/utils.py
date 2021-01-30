"""Collections of util functions and classes."""

import numpy as np
import tensorflow as tf
from collections import defaultdict
from typing import List


def outer(x: tf.Tensor, y: tf.Tensor):
  return tf.expand_dims(x, axis=-1) * tf.expand_dims(y, axis=-2)


def random(shape: List[int]):
  return tf.random.uniform(shape=shape, minval=0., maxval=1.)


def expect(x: tf.Tensor):
  return tf.reduce_mean(x, axis=0)


def create_variable(name: str,
                    shape: List[int],
                    initializer: tf.initializers.Initializer,
                    dtype: str = 'float32',
                    trainable: bool = True):
  init_value = initializer(shape, dtype)
  return tf.Variable(init_value, trainable=trainable, name=name)


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
