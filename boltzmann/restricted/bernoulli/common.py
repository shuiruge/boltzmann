"""Common part of the Bernoulli restricted Boltzmann machines."""

import tensorflow as tf
from boltzmann.utils import expect, random
from boltzmann.restricted.base import Initializer, Distribution


# TODO: Add sparsity
class GlorotInitializer(Initializer):

  def __init__(self,
               samples: tf.Tensor,
               eps: float = 1e-8,
               seed: int = None):
    self.samples = samples
    self.eps = eps
    self.seed = seed

  @property
  def kernel(self):
    # return tf.initializers.glorot_normal(seed=self.seed)

    p = expect(self.samples)

    # TODO: test!
    def initializer(shape, dtype):
      ambient_size, latent_size = shape
      # shape: [ambient_size]
      stddev = 2 / (
          (p + self.eps) * (1 - p + self.eps) * ambient_size
          + (0.5) * (1 - 0.5) * latent_size)
      # shape: [ambient_size, latent_size]
      stddev = tf.expand_dims(stddev, axis=1)
      return stddev * tf.random.truncated_normal(
          shape, dtype=dtype, seed=self.seed)

    return initializer

  @property
  def ambient_bias(self):
    """C.f. Hinton (2012)."""
    p = expect(self.samples)

    def initializer(_, dtype):
      b = tf.math.log(p + self.eps) - tf.math.log(1 - p + self.eps)
      return tf.cast(b, dtype)

    return initializer

  @property
  def latent_bias(self):
    """C.f. Hinton (2012)."""
    return tf.initializers.zeros()


class Bernoulli(Distribution):

  def __init__(self, prob: tf.Tensor):
    self.prob = prob

  def sample(self, seed: int = None) -> tf.Tensor:
    rand = random(self.prob.shape, seed=seed)
    y = tf.where(rand <= self.prob, 1, 0)
    return tf.cast(y, self.prob.dtype)

  @property
  def prob_argmax(self) -> tf.Tensor:
    y = tf.where(self.prob >= 0.5, 1, 0)
    return tf.cast(y, self.prob.dtype)
