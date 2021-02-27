import abc
from boltzmann.restricted.base import Districution


class Initializer(abc.ABC):

  @abc.abstractproperty
  def ambient_ambient_kernel(self):
    return NotImplemented

  @abc.abstractproperty
  def ambient_bias(self):
    return NotImplemented

  @abc.abstractproperty
  def latent_latent_kernel(self):
    return NotImplemented

  @abc.abstractproperty
  def latent_bias(self):
    return NotImplemented

  @abc.abstractproperty
  def ambient_latent_kernel(self):
    return NotImplemented


class State:

  def __init__(self, ambient: Districution, latent: Districution):
    self.ambient = ambient
    self.latent = latent


class BoltzmannMachine(abc.ABC):

  @abc.abstractmethod
  def update_state(self, state: State) -> State:
    return NotImplemented
