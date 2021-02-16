"""Dense Bernoulli restricted Boltzmann machine."""

import tensorflow as tf
from boltzmann.utils import (
    create_variable, get_sparsity_constraint, inner)
from boltzmann.restricted.base import (
    Initializer, RestrictedBoltzmannMachine)
from boltzmann.restricted.bernoulli import Bernoulli


class DenseBernoulliRBM(RestrictedBoltzmannMachine):

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
    a = h @ tf.transpose(W) + v
    return Bernoulli(tf.sigmoid(a))


def get_energy(rbm: DenseBernoulliRBM,
               ambient: tf.Tensor,
               latent: tf.Tensor):
  x, h = ambient, latent
  W, b, v = rbm.kernel, rbm.latent_bias, rbm.ambient_bias
  energy: tf.Tensor = (
      - tf.reduce_sum(x @ W * h, axis=-1)
      - tf.reduce_mean(h * b, axis=-1)
      - tf.reduce_mean(x * v, axis=-1)
  )
  return energy


def get_free_energy(rbm: DenseBernoulliRBM, ambient: tf.Tensor):
  W, b, v, x = rbm.kernel, rbm.latent_bias, rbm.ambient_bias, ambient
  free_energy: tf.Tensor = (
      -inner(v, x)
      - tf.reduce_sum(tf.math.softplus(x @ W + b), axis=-1)
  )
  return free_energy


def initialize_fantasy_latent(rbm: DenseBernoulliRBM,
                              num_samples: int,
                              prob: float = 0.5,
                              seed: int = None):
  p = prob * tf.ones([num_samples, rbm.latent_size])
  return Bernoulli(p).sample(seed=seed)


class LatentIncrementingInitializer(Initializer):

  def __init__(self, base_rbm: RestrictedBoltzmannMachine, increment: int):
    self.base_rbm = base_rbm
    self.increment = increment

  @property
  def kernel(self):

    def initializer(*_):
      return tf.concat(
          [
              self.base_rbm.kernel,
              tf.zeros([self.base_rbm.ambient_size, self.increment]),
          ],
          axis=1)

    return initializer

  @property
  def ambient_bias(self):

    def initializer(*_):
      return self.base_rbm.ambient_bias

    return initializer

  @property
  def latent_bias(self):

    def initializer(*_):
      return tf.concat(
          [
              self.base_rbm.latent_bias,
              tf.zeros([self.increment]),
          ],
          axis=0)

    return initializer


def enlarge_latent(base_rbm, base_fantasy_latent, increment):
  seed = base_rbm.seed
  rbm = DenseBernoulliRBM(
      ambient_size=base_rbm.ambient_size,
      latent_size=(base_rbm.latent_size + increment),
      initializer=LatentIncrementingInitializer(base_rbm, increment),
      seed=seed)
  # prob = tf.reduce_mean(base_fantasy_latent)  # TODO: needs discussion.
  prob = 0.5
  fantasy_latent = tf.concat(
      [
          base_fantasy_latent,
          initialize_fantasy_latent(
              increment, base_fantasy_latent.shape[0], prob=prob, seed=seed),
      ],
      axis=1)
  return rbm, fantasy_latent


def get_reconstruction_error(rbm, DenseBernoulliRBM, real_ambient: tf.Tensor):
  real_latent = rbm.get_latent_given_ambient(real_ambient).prob_argmax
  recon_ambient = rbm.get_ambient_given_latent(real_latent).prob_argmax
  recon_error: tf.Tensor = tf.reduce_mean(
      tf.cast(recon_ambient != real_ambient, 'float32'))
  return recon_error
