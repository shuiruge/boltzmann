"""Bernoulli restricted Boltzmann machine."""

import tensorflow as tf
from typing import List
from boltzmann.base import Initializer, Distribution
from boltzmann.utils import History, expect, outer, random, create_variable


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


class BernoulliRBM:

  def __init__(self,
               ambient_size: int,
               latent_size: int,
               initializer: Initializer):
    self.ambient_size = ambient_size
    self.latent_size = latent_size
    self.initializer = initializer

    self.kernel = create_variable(
        name='kernel',
        shape=[ambient_size, latent_size],
        initializer=self.initializer.kernel,
    )
    self.latent_bias = create_variable(
        name='latent_bias',
        shape=[latent_size],
        initializer=self.initializer.latent_bias,
    )
    self.ambient_bias = create_variable(
        name='ambient_bias',
        shape=[ambient_size],
        initializer=self.initializer.ambient_bias,
    )


class Bernoulli(Distribution):

  def __init__(self, prob: tf.Tensor):
    self.prob = prob
  
  def sample(self):
    y = tf.where(random(self.prob.shape) <= self.prob, 1, 0)
    return tf.cast(y, self.prob.dtype)
  
  @property
  def prob_argmax(self):
    y = tf.where(self.prob >= 0.5, 1, 0)
    return tf.cast(y, self.prob.dtype)


def latent_given_ambient(rbm: BernoulliRBM, ambient: tf.Tensor):
  W, b, x = rbm.kernel, rbm.latent_bias, ambient
  a = x @ W + b
  return Bernoulli(tf.sigmoid(a))


def ambient_given_latent(rbm: BernoulliRBM, latent: tf.Tensor):
  W, v, h = rbm.kernel, rbm.ambient_bias, latent
  a = h @ tf.transpose(W) + v
  return Bernoulli(tf.sigmoid(a))


def relax(rbm: BernoulliRBM, ambient: tf.Tensor, max_iter: int, tol: float):
  for step in tf.range(max_iter):
    latent = latent_given_ambient(rbm, ambient).prob_argmax
    new_ambient = ambient_given_latent(rbm, latent).prob_argmax
    if tf.reduce_max(tf.abs(new_ambient - ambient)) < tol:
      break
    ambient = new_ambient
  return ambient, step


def get_energy(rbm: BernoulliRBM, ambient: tf.Tensor, latent: tf.Tensor):
  x, h = ambient, latent
  W, b, v = rbm.kernel, rbm.latent_bias, rbm.ambient_bias
  energy = (
      - tf.reduce_sum(x @ W * h, axis=-1)
      - tf.reduce_mean(h * b, axis=-1)
      - tf.reduce_mean(x * v, axis=-1)
  )
  return energy


def init_fantasy_latent(rbm: BernoulliRBM, num_samples: int):
  p = 0.5 * tf.ones([num_samples, rbm.latent_size])
  return Bernoulli(p).sample()


def get_grads_and_vars(rbm: BernoulliRBM,
                       real_ambient: tf.Tensor,
                       fantasy_latent: tf.Tensor):
  real_latent = latent_given_ambient(rbm, real_ambient).sample()
  fantasy_ambient_prob = ambient_given_latent(rbm, fantasy_latent).prob

  grad_kernel = (
      expect(outer(fantasy_ambient_prob, fantasy_latent))
      - expect(outer(real_ambient, real_latent))
  )
  grad_latent_bias = expect(fantasy_latent) - expect(real_latent)
  grad_ambient_bias = expect(fantasy_ambient_prob) - expect(real_ambient)

  return [
      (grad_kernel, rbm.kernel),
      (grad_latent_bias, rbm.latent_bias),
      (grad_ambient_bias, rbm.ambient_bias),
  ]


def contrastive_divergence(rbm: BernoulliRBM,
                           fantasy_latent: tf.Tensor,
                           mc_steps: int):
  for _ in tf.range(mc_steps):
    fantasy_ambient = ambient_given_latent(rbm, fantasy_latent).sample()
    fantasy_latent = latent_given_ambient(rbm, fantasy_ambient).sample()
  return fantasy_latent


def train(rbm: BernoulliRBM,
          optimizer: tf.optimizers.Optimizer,
          dataset: tf.data.Dataset,
          fantasy_latent: tf.Tensor,
          mc_steps: int = 1,
          history: History = None):
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
  real_latent = latent_given_ambient(rbm, real_ambient).prob_argmax
  recon_ambient = ambient_given_latent(rbm, real_latent).prob_argmax

  mean_energy = tf.reduce_mean(get_energy(rbm, real_ambient, real_latent))
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


if __name__ == '__main__':

  from boltzmann.data.mnist import load_mnist

  image_size = (16, 16)
  (X, _), _ = load_mnist(image_size=image_size, binarize=True, minval=0, maxval=1)

  ambient_size = image_size[0] * image_size[1]
  latent_size = 64
  batch_size = 128
  dataset = tf.data.Dataset.from_tensor_slices(X)
  dataset = dataset.shuffle(10000).repeat(10).batch(batch_size)
  rbm = BernoulliRBM(ambient_size, latent_size, HintonInitializer(X))
  fantasy_latent = init_fantasy_latent(rbm, batch_size)
  optimizer = tf.optimizers.Adam()
  history = History()
  fantasy_latent = train(rbm, optimizer, dataset, fantasy_latent,
                         history=history)
