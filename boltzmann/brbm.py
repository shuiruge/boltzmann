"""Bernoulli restricted Boltzmann machine."""

import tensorflow as tf
from boltzmann.base import (
    Initializer, Distribution, contrastive_divergence, get_grads_and_vars)
from boltzmann.utils import History, expect, random, create_variable


class GlorotInitializer(Initializer):

  def __init__(self, samples: tf.Tensor, eps: float = 1e-8):
    self.samples = samples
    self.eps = eps

  @property
  def kernel(self):
    return tf.initializers.glorot_normal()

  @property
  def ambient_bias(self):

    def initializer(_, dtype):
      b = 1 / (expect(self.samples) + self.eps)
      return tf.cast(b, dtype)

    return initializer

  @property
  def latent_bias(self):
    return tf.initializers.zeros()


class HintonInitializer(Initializer):

  def __init__(self, samples: tf.Tensor, eps: float = 1e-8):
    self.samples = samples
    self.eps = eps

  @property
  def kernel(self):
    return tf.random_normal_initializer(stddev=1e-2)

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

  def sample(self) -> tf.Tensor:
    y = tf.where(random(self.prob.shape) <= self.prob, 1, 0)
    return tf.cast(y, self.prob.dtype)

  @property
  def prob_argmax(self) -> tf.Tensor:
    y = tf.where(self.prob >= 0.5, 1, 0)
    return tf.cast(y, self.prob.dtype)


class BernoulliRBM:

  def __init__(self,
               ambient_size: int,
               latent_size: int,
               initializer: Initializer):
    self.ambient_size = ambient_size
    self.latent_size = latent_size
    self.initializer = initializer

    self._kernel = create_variable(
        name='kernel',
        shape=[ambient_size, latent_size],
        initializer=self.initializer.kernel,
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

  def get_energy(self, ambient: tf.Tensor, latent: tf.Tensor):
    x, h = ambient, latent
    W, b, v = self.kernel, self.latent_bias, self.ambient_bias
    energy: tf.Tensor = (
        - tf.reduce_sum(x @ W * h, axis=-1)
        - tf.reduce_mean(h * b, axis=-1)
        - tf.reduce_mean(x * v, axis=-1)
    )
    return energy


def init_fantasy_latent(rbm: BernoulliRBM, num_samples: int):
  p = 0.5 * tf.ones([num_samples, rbm.latent_size])
  return Bernoulli(p).sample()


def train(rbm: BernoulliRBM,
          optimizer: tf.optimizers.Optimizer,
          dataset: tf.data.Dataset,
          fantasy_latent: tf.Tensor,
          mc_steps: int = 1,
          history: History = None):
  """Returns the final fantasy latent."""
  for step, real_ambient in enumerate(dataset):
    grads_and_vars = get_grads_and_vars(rbm, real_ambient, fantasy_latent)
    optimizer.apply_gradients(grads_and_vars)
    fantasy_latent = contrastive_divergence(rbm, fantasy_latent, mc_steps)

    if history is not None and step % 10 == 0:
      log_and_print_internal_information(
          history, rbm, step, real_ambient, fantasy_latent)

  return fantasy_latent


def log_and_print_internal_information(
        history, rbm, step, real_ambient, fantasy_latent):
  real_latent = rbm.get_latent_given_ambient(real_ambient).prob_argmax
  recon_ambient = rbm.get_ambient_given_latent(real_latent).prob_argmax

  mean_energy = tf.reduce_mean(rbm.get_energy(real_ambient, real_latent))
  recon_error = tf.reduce_mean(
      tf.cast(recon_ambient == real_ambient, 'float32'))
  latent_on_ratio = tf.reduce_mean(real_latent)

  def stats(x, name):
    mean, var = tf.nn.moments(x, axes=range(len(x.shape)))
    std = tf.sqrt(var)
    history.log(step, f'{name}', f'{mean:.5f} ({std:.5f})')

  history.log(step, 'mean energy', mean_energy)
  history.log(step, 'recon error', recon_error)
  history.log(step, 'latent-on ratio', latent_on_ratio)

  stats(rbm.kernel, 'kernel')
  stats(rbm.ambient_bias, 'ambient bias')
  stats(rbm.latent_bias, 'latent bias')

  print(history.show(step))
