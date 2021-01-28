"""Defines interfaces."""

import abc
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
