import tensorflow as tf
from boltzmann.utils import create_variable


class BernoulliBoltzmannMachine:

  def __init__(self, ambient_size: int, latent_size: int):
    self.ambient_size = ambient_size
    self.latent_size = latent_size

    self.latent_latent_kernel = create_variable('latent_latent_kernel', constraint=...)