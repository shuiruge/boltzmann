"""Common part of the Gaussian restricted Boltzmann machines."""

import tensorflow as tf
from boltzmann.restricted.base import Initializer, Distribution


class GlorotInitializer(Initializer):

  def __init__(self, samples: tf.Tensor, eps: float = 1e-8, seed: int = None):
    self.samples = samples
    self.eps = eps
    self.seed = seed

  @property
  def kernel(self):
    return tf.initializers.glorot_normal(seed=self.seed)

  # TODO
  @property
  def ambient_bias(self):

    def initializer(*_):
      return NotImplemented

    return initializer

  @property
  def latent_bias(self):
    """C.f. Hinton (2012)."""
    return tf.initializers.zeros()


class Gaussian(Distribution):

  def __init__(self, mean: tf.Tensor, stddev: tf.Tensor):
    self.mean = mean
    self.stddev = stddev

  def sample(self, seed: int = None) -> tf.Tensor:
    return tf.random.truncated_normal(shape=self.mu.shape,
                                      mean=self.mean,
                                      stddev=self.stddev,
                                      dtype=self.mu.dtype,
                                      seed=seed)

  @property
  def prob_argmax(self) -> tf.Tensor:
    return self.mean
