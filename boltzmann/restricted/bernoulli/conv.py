"""Convolutional Bernoulli restricted Boltzmann machine."""

import tensorflow as tf
from typing import List
from boltzmann.restricted.base import Initializer, RestrictedBoltzmannMachine


# TODO
class ConvBernoulliRBM(RestrictedBoltzmannMachine):

  def __init__(self,
               ambient_shape: List[int],
               latent_shape: List[int],
               initializer: Initializer,
               seed: int = None):
    self.ambient_shape = ambient_shape
    self.latent_shape = latent_shape
    self.initializer = initializer
    self.seed = seed

    self._kernel = NotImplemented
    self._latent_bias = NotImplemented
    self._ambient_bias = NotImplemented

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
    return NotImplemented

  def get_ambient_given_latent(self, latent: tf.Tensor):
    return NotImplemented
