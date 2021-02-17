"""Dense Gaussian restricted Boltzmann machine."""

import tensorflow as tf
from boltzmann.restricted.base import Initializer, RestrictedBoltzmannMachine
from boltzmann.restricted.gaussian.common import Gaussian
from boltzmann.restricted.bernoulli.common import Bernoulli
from boltzmann.utils import create_variable, get_sparsity_constraint


class DenseGaussianRBM(RestrictedBoltzmannMachine):
  """Gaussian restricted Boltzmann machine.

  As ref[1] suggests, ambient shall be standardized first s.t. the stddev is
  unit.

  References
  ----------
  1. Hinton (2012), section 13.2.
  """

  def __init__(self,
               ambient_size: int,
               latent_size: int,
               initializer: Initializer,
               sparsity: float = 0,
               seed: int = None):
    self.ambient_size = ambient_size
    self.latent_size = latent_size
    self.initializer = initializer
    self.sparsity = sparsity
    self.seed = seed

    self._kernel = create_variable(
        name='kernel',
        shape=[ambient_size, latent_size],
        initializer=self.initializer.kernel,
        constraint=get_sparsity_constraint(sparsity, seed),
    )
    self._latent_bias = create_variable(
        name='latent_bias',
        shape=[latent_size],
        initializer=self.initializer.latent_bias,
    )
    self._ambient_bias = create_variable(
        name='ambient_bias',
        shape=[ambient_size],
        initializer=self.initializer.ambient_bias,
    )

  @property
  def kernel(self):
    return self._kernel

  @property
  def ambient_bias(self):
    return self._ambient_bias

  @property
  def latent_bias(self):
    return self._latent_bias

  def get_latent_given_ambient(self, ambient: tf.Tensor):
    W, b, x = self.kernel, self.latent_bias, ambient
    a = x @ W + b
    return Bernoulli(tf.sigmoid(a))

  def get_ambient_given_latent(self, latent: tf.Tensor):
    W, v, h = self.kernel, self.ambient_bias, latent
    mean = h @ tf.transpose(W) + v
    stddev = tf.ones_like(mean)
    return Gaussian(mean, stddev)


def get_reconstruction_error(rbm: DenseGaussianRBM, real_ambient: tf.Tensor):
  real_latent = rbm.get_latent_given_ambient(real_ambient).prob_argmax
  recon_ambient = rbm.get_ambient_given_latent(real_latent).prob_argmax
  recon_error: tf.Tensor = tf.reduce_mean(
      tf.square(recon_ambient - real_ambient))
  return recon_error
