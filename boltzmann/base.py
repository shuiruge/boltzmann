"""Defines interfaces."""

import abc
from boltzmann.utils import expect, outer
import tensorflow as tf


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

  @abc.abstractmethod
  def get_energy(self, ambient: tf.Tensor, latent: tf.Tensor) -> tf.Tensor:
    return NotImplemented


def relax(rbm: RestrictedBoltzmannMachine,
          ambient: tf.Tensor,
          max_iter: int,
          tol: float):
  for step in tf.range(max_iter):
    latent = rbm.get_latent_given_ambient(ambient).prob_argmax
    new_ambient = rbm.get_ambient_given_latent(latent).prob_argmax
    if tf.reduce_max(tf.abs(new_ambient - ambient)) < tol:
      break
    ambient = new_ambient
  return ambient, step


def contrastive_divergence(rbm: RestrictedBoltzmannMachine,
                           fantasy_latent: tf.Tensor,
                           mc_steps: int):
  for _ in tf.range(mc_steps):
    fantasy_ambient = rbm.get_ambient_given_latent(fantasy_latent).sample()
    fantasy_latent = rbm.get_latent_given_ambient(fantasy_ambient).sample()
  return fantasy_latent


def get_grads_and_vars(rbm: RestrictedBoltzmannMachine,
                       real_ambient: tf.Tensor,
                       fantasy_latent: tf.Tensor):
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
