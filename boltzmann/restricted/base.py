"""Defines interfaces of restricted Boltzmann machine and related."""

import abc
import tensorflow as tf
from typing import List
from copy import deepcopy
from boltzmann.utils import History, inplace, expect, outer, quantize_tensor


class Initializer(abc.ABC):

  @abc.abstractproperty
  def kernel(self) -> tf.initializers.Initializer:
    return NotImplemented

  @abc.abstractproperty
  def ambient_bias(self) -> tf.initializers.Initializer:
    return NotImplemented

  @abc.abstractproperty
  def latent_bias(self) -> tf.initializers.Initializer:
    return NotImplemented


class Distribution(abc.ABC):

  @abc.abstractmethod
  def sample(self) -> tf.Tensor:
    return NotImplemented

  @abc.abstractproperty
  def prob_argmax(self) -> tf.Tensor:
    return NotImplemented


class RestrictedBoltzmannMachine(abc.ABC):

  @abc.abstractproperty
  def kernel(self) -> tf.Tensor:
    return NotImplemented

  @abc.abstractproperty
  def ambient_bias(self) -> tf.Tensor:
    return NotImplemented

  @abc.abstractproperty
  def latent_bias(self) -> tf.Tensor:
    return NotImplemented

  @abc.abstractmethod
  def get_latent_given_ambient(self, ambient: tf.Tensor) -> Distribution:
    return NotImplemented

  @abc.abstractmethod
  def get_ambient_given_latent(self, latent: tf.Tensor) -> Distribution:
    return NotImplemented


def relax(rbm: RestrictedBoltzmannMachine,
          ambient: tf.Tensor,
          max_step: int,
          tolerance: float):
  """Evolves the dynamics until the two adjacent ambients is the same, and
  returns the final ambient and the final step of evolution.

  The maximum step of the evoluation is `max_step`, until then stop the
  evolution, regardless whether it has been relaxed or not.

  The word "same" means that the L-infinity norm of the difference is smaller
  than the `tolerance`.
  """
  step = 0
  for _ in tf.range(max_step):
    latent = rbm.get_latent_given_ambient(ambient).prob_argmax
    new_ambient = rbm.get_ambient_given_latent(latent).prob_argmax
    if infinity_norm(new_ambient - ambient) < tolerance:
      break
    ambient = new_ambient
    step += 1
  return ambient, step


def infinity_norm(x: tf.Tensor):
  norm: tf.Tensor = tf.reduce_max(tf.abs(x))
  return norm


def contrastive_divergence(rbm: RestrictedBoltzmannMachine,
                           fantasy_latent: tf.Tensor,
                           mc_steps: int):
  """Returns the final fantasy latent."""
  for _ in tf.range(mc_steps):
    fantasy_ambient = rbm.get_ambient_given_latent(fantasy_latent).sample()
    fantasy_latent = rbm.get_latent_given_ambient(fantasy_ambient).sample()
  return fantasy_latent


def get_grads_and_vars(rbm: RestrictedBoltzmannMachine,
                       real_ambient: tf.Tensor,
                       fantasy_latent: tf.Tensor):
  """For applying `tf.optimizers.Optimizer.apply_gradients` method."""
  real_latent = rbm.get_latent_given_ambient(real_ambient).sample()
  fantasy_ambient = rbm.get_ambient_given_latent(fantasy_latent).sample()

  grad_kernel = (
      expect(outer(fantasy_ambient, fantasy_latent))
      - expect(outer(real_ambient, real_latent))
  )
  grad_latent_bias = expect(fantasy_latent) - expect(real_latent)
  grad_ambient_bias = expect(fantasy_ambient) - expect(real_ambient)

  return [
      (grad_kernel, rbm.kernel),
      (grad_latent_bias, rbm.latent_bias),
      (grad_ambient_bias, rbm.ambient_bias),
  ]


class Callback(abc.ABC):
  """For `train`."""

  @abc.abstractmethod
  def __call__(self,
               step: int,
               real_ambient: tf.Tensor,
               fantasy_latent: tf.Tensor,
               ) -> None:
    return NotImplemented


@inplace('rbm, e.t.c.')
def train(rbm: RestrictedBoltzmannMachine,
          optimizer: tf.optimizers.Optimizer,
          dataset: tf.data.Dataset,
          fantasy_latent: tf.Tensor,
          mc_steps: int = 1,
          callbacks: List[Callback] = None):
  """Returns the final fantasy latent."""
  for step, real_ambient in enumerate(dataset):
    grads_and_vars = get_grads_and_vars(rbm, real_ambient, fantasy_latent)
    optimizer.apply_gradients(grads_and_vars)
    fantasy_latent = contrastive_divergence(rbm, fantasy_latent, mc_steps)

    if callbacks is None:
      callbacks = []
    for callback in callbacks:
      callback(step, real_ambient, fantasy_latent)

  return fantasy_latent


class LogInternalInformation(Callback):

  def __init__(self,
               rbm: RestrictedBoltzmannMachine,
               log_step: int,
               verbose: bool):
    self.rbm = rbm
    self.log_step = log_step
    self.verbose = verbose

    self.history = History()

  def __call__(self,
               step: int,
               real_ambient: tf.Tensor,
               fantasy_latent: tf.Tensor):
    if step % self.log_step != 0:
      return

    def stats(x, name):
      mean, var = tf.nn.moments(x, axes=range(len(x.shape)))
      std = tf.sqrt(var)
      self.history.log(step, f'{name}', f'{mean:.5f} ({std:.5f})')

    real_latent = self.rbm.get_latent_given_ambient(real_ambient).prob_argmax
    stats(real_latent, 'real latent')
    stats(self.rbm.kernel, 'kernel')
    stats(self.rbm.ambient_bias, 'ambient bias')
    stats(self.rbm.latent_bias, 'latent bias')

    if self.verbose:
      print(self.history.show(step))


def quantize(rbm: RestrictedBoltzmannMachine, precision: float):
  quantized_rbm = deepcopy(rbm)
  for attr in ('kernel', 'ambient_bias', 'latent_bias'):
    getattr(quantized_rbm, attr).assign(
        quantize_tensor(getattr(rbm, attr), precision))
  return quantized_rbm
