import tensorflow as tf
from boltzmann.generic.base import Initializer, State, BoltzmannMachine
from boltzmann.utils import (
    SymmetricDiagonalVanishingConstraint, create_variable)
from boltzmann.restricted.bernoulli.common import Bernoulli


class BernoulliBoltzmannMachine(BoltzmannMachine):

  def __init__(self,
               ambient_size: int,
               latent_size: int,
               initializer: Initializer,
               seed: int = None):
    self.ambient_size = ambient_size
    self.latent_size = latent_size
    self.initializer = initializer
    self.seed = seed

    self.ambient_ambient_kernel = create_variable(
        name='ambient_ambient_kernel',
        shape=[ambient_size, ambient_size],
        initializer=initializer.ambient_ambient_kernel,
        constraint=SymmetricDiagonalVanishingConstraint(),
    )
    self.ambient_bias = create_variable(
        name='ambient_bias',
        shape=[ambient_size],
        initializer=initializer.ambient_bias,
    )
    self.latent_latent_kernel = create_variable(
        name='latent_latent_kernel',
        shape=[latent_size, latent_size],
        initializer=initializer.latent_latent_kernel,
        constraint=SymmetricDiagonalVanishingConstraint(),
    )
    self.latent_bias = create_variable(
        name='latent_bias',
        shape=[latent_size],
        initializer=initializer.latent_bias,
    )
    self.ambient_latent_kernel = create_variable(
        name='ambient_latent_kernel',
        shape=[ambient_size, latent_size],
        initializer=initializer.ambient_latent_kernel,
    )

  def update_state(self, state: State):
    # abbreviations
    v, h = state.ambient, state.latent
    W = self.ambient_latent_kernel
    L = self.ambient_ambient_kernel
    J = self.latent_latent_kernel
    bv = self.ambient_bias
    bh = self.latent_bias

    # get ambient given state
    ambient = Bernoulli(tf.sigmoid(
        h @ tf.transpose(W) + v @ L + bv))

    # get latent given state
    latent = Bernoulli(tf.sigmoid(
        v @ W + h @ J + bh))

    return State(ambient, latent)
