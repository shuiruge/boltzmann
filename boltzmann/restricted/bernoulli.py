"""Bernoulli restricted Boltzmann machine."""

import tensorflow as tf
from boltzmann.utils import (
    History, expect, inner, random, create_variable, get_sparsity_constraint)
from boltzmann.restricted.base import (
    Callback, Initializer, Distribution, RestrictedBoltzmannMachine)


class GlorotInitializer(Initializer):

  def __init__(self, samples: tf.Tensor, eps: float = 1e-8, seed: int = None):
    self.samples = samples
    self.eps = eps
    self.seed = seed

  @property
  def kernel(self):
    return tf.initializers.glorot_normal(seed=self.seed)

  @property
  def ambient_bias(self):
    p = expect(self.samples)

    def initializer(_, dtype):
      b = tf.math.log(p + self.eps) - tf.math.log(1 - p + self.eps)
      return tf.cast(b, dtype)

    return initializer

  @property
  def latent_bias(self):
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


class BernoulliRBM:

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


def get_energy(rbm: RestrictedBoltzmannMachine,
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


def get_free_energy(rbm: RestrictedBoltzmannMachine, ambient: tf.Tensor):
  W, b, v, x = rbm.kernel, rbm.latent_bias, rbm.ambient_bias, ambient
  free_energy: tf.Tensor = (
      -inner(v, x)
      - tf.reduce_sum(tf.math.softplus(x @ W + b), axis=-1)
  )
  return free_energy


def init_fantasy_latent(rbm: BernoulliRBM,
                        num_samples: int,
                        seed: int = None):
  p = 0.5 * tf.ones([num_samples, rbm.latent_size])
  return Bernoulli(p).sample(seed=seed)


class LogAndPrintInternalInformation(Callback):

  def __init__(self, rbm: RestrictedBoltzmannMachine, log_step: int):
    self.rbm = rbm
    self.log_step = log_step

    self.history = History()

  def __call__(self,
               step: int,
               real_ambient: tf.Tensor,
               fantasy_latent: tf.Tensor):
    if step % self.log_step != 0:
      return

    real_latent = self.rbm.get_latent_given_ambient(real_ambient).prob_argmax
    recon_ambient = self.rbm.get_ambient_given_latent(real_latent).prob_argmax

    mean_energy = tf.reduce_mean(
        get_energy(self.rbm, real_ambient, real_latent))
    recon_error = tf.reduce_mean(
        tf.cast(recon_ambient != real_ambient, 'float32'))
    latent_on_ratio = tf.reduce_mean(real_latent)
    mean_free_energy = tf.reduce_mean(get_free_energy(self.rbm, real_ambient))

    def stats(x, name):
      mean, var = tf.nn.moments(x, axes=range(len(x.shape)))
      std = tf.sqrt(var)
      self.history.log(step, f'{name}', f'{mean:.5f} ({std:.5f})')

    self.history.log(step, 'mean energy', mean_energy)
    self.history.log(step, 'recon error', recon_error)
    self.history.log(step, 'latent-on ratio', latent_on_ratio)
    self.history.log(step, 'mean free energy', mean_free_energy)

    stats(self.rbm.kernel, 'kernel')
    stats(self.rbm.ambient_bias, 'ambient bias')
    stats(self.rbm.latent_bias, 'latent bias')

    print(self.history.show(step))
