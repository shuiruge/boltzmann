import tensorflow as tf
from typing import Optional

import boltzmann.generic.base as B
from boltzmann.generic.base import (
    Distribution, Initializer, State, BoltzmannMachine, Callback)
from boltzmann.utils import (
    History, SymmetricDiagonalVanishingConstraint, create_variable,
    outer, random, expect, infinity_norm)


def get_batch_size(batch_of_data: tf.Tensor):
  """If the tensor `batch_of_data` represents a batch of data, then always
  assumes that the first, and only the first, axis is for batch."""
  return batch_of_data.shape[0]


class Bernoulli(Distribution):

  def __init__(self, prob: tf.Tensor):
    self.prob = prob

  def sample(self, seed: int):
    rand = random(self.prob.shape, seed)
    y = tf.where(rand <= self.prob, 1, 0)
    return tf.cast(y, self.prob.dtype)

  @property
  def prob_argmax(self):
    y = tf.where(self.prob >= 0.5, 1, 0)
    return tf.cast(y, self.prob.dtype)


class HintonInitializer(Initializer):

  def __init__(self,
               samples: tf.Tensor,
               eps: float = 1e-8,
               seed: int = 42):
    self.samples = samples
    self.eps = eps
    self.seed = seed

  @property
  def ambient_ambient_kernel(self):
    return tf.initializers.zeros()

  @property
  def ambient_bias(self):
    p = expect(self.samples)

    def initializer(_, dtype):
      b = tf.math.log(p + self.eps) - tf.math.log(1 - p + self.eps)
      return tf.cast(b, dtype)

    return initializer

  @property
  def latent_latent_kernel(self):
    return tf.initializers.zeros()

  @property
  def latent_bias(self):
    return tf.initializers.zeros()

  @property
  def ambient_latent_kernel(self):
    return tf.initializers.zeros()


class BernoulliBoltzmannMachine(BoltzmannMachine):

  def __init__(self,
               ambient_size: int,
               latent_size: int,
               initializer: Initializer,
               max_step: int = 10,
               tolerance: float = 1e-1,
               connect_ambient_to_ambient: bool = True,
               connect_latent_to_latent: bool = True,
               use_latent_bias: bool = True,
               activate_ratio: float = 1,
               debug_mode: bool = False,
               seed: int = 42):
    self.ambient_size = ambient_size
    self.latent_size = latent_size
    self.initializer = initializer
    self.seed = seed
    self.max_step = max_step
    self.tolerance = tolerance
    self.connect_ambient_to_ambient = connect_ambient_to_ambient
    self.connect_latent_to_latent = connect_latent_to_latent
    self.use_latent_bias = use_latent_bias
    self.activate_ratio = activate_ratio
    self.debug_mode = debug_mode
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

  def get_config(self):
    return {
        'ambient_size': self.ambient_size,
        'latent_size': self.latent_size,
        'initializer': self.initializer,
        'max_step': self.max_step,
        'tolerance': self.tolerance,
        'connect_ambient_to_ambient': self.connect_ambient_to_ambient,
        'connect_latent_to_latent': self.connect_latent_to_latent,
        'use_latent_bias': self.use_latent_bias,
        'activate_ratio': self.activate_ratio,
        'debug_mode': self.debug_mode,
        'seed': self.seed,
    }

  @property
  def params_and_ops(self):
    result = []
    result += [(
        self.ambient_latent_kernel,
        lambda state: outer(state.ambient, state.latent),
    )]
    result += [(
        self.ambient_bias,
        lambda state: state.ambient,
    )]
    if self.use_latent_bias:
      result += [(
          self.latent_bias,
          lambda state: state.latent,
      )]
    if self.connect_ambient_to_ambient:
      result += [(
          self.ambient_ambient_kernel,
          lambda state: outer(state.ambient, state.ambient),
      )]
    if self.connect_latent_to_latent:
      result += [(
          self.latent_latent_kernel,
          lambda state: outer(state.latent, state.latent),
      )]
    return result

  def gibbs_sampling(self, state: State):
    # abbreviations
    v, h = state.ambient, state.latent
    W = self.ambient_latent_kernel
    L = self.ambient_ambient_kernel
    J = self.latent_latent_kernel
    bv = self.ambient_bias
    bh = self.latent_bias

    # get ambient given state
    v = Bernoulli(
        tf.sigmoid(h @ tf.transpose(W) + v @ L + bv)
    ).sample(self.seed)

    # get latent given state
    h = Bernoulli(
        tf.sigmoid(v @ W + h @ J + bh)
    ).sample(self.seed)

    return State(v, h)

  def activate(self, state: State):
    if self.activate_ratio < 1:
      return stochastic_activate(self, state, self.activate_ratio, self.seed)
    else:
      return deterministic_activate(self, state, None, None)

  def get_latent_given_ambient(self, ambient: tf.Tensor):
    latent, final_step = mean_field_approx(
        self, ambient, self.max_step, self.tolerance, self.seed)

    if self.debug_mode and final_step == self.max_step:
      warning_message = (
          'Failed in getting latent via mean field approximation. '
          'Try either increasing `max_step` or decreasing `tolerance`.')
      tf.print('[WARNING]', warning_message)

    return latent


def deterministic_activate(bm: BernoulliBoltzmannMachine,
                           state: State,
                           ambient_mask: Optional[tf.Tensor],
                           latent_mask: Optional[tf.Tensor]):
  """Activates ambient units and then latent in order, once."""
  # abbreviations
  v, h = state.ambient, state.latent
  W = bm.ambient_latent_kernel
  L = bm.ambient_ambient_kernel
  J = bm.latent_latent_kernel
  bv = bm.ambient_bias
  bh = bm.latent_bias

  # get ambient given state
  new_v = Bernoulli(
      tf.sigmoid(h @ tf.transpose(W) + v @ L + bv)
  ).prob_argmax

  if ambient_mask is None:
    v = new_v
  else:
    v = tf.where(ambient_mask > 0, new_v, v)

  # get latent given state
  new_h = Bernoulli(
      tf.sigmoid(v @ W + h @ J + bh)
  ).prob_argmax

  if latent_mask is None:
    h = new_h
  else:
    h = tf.where(latent_mask > 0, new_h, h)

  return State(v, h)


def stochastic_activate(bm: BernoulliBoltzmannMachine,
                        state: State,
                        activate_ratio: float,
                        seed: int):
  """Activates ambient units and then latent in order iteratively, with
  activating ratio `activate_ratio` in each activation, untill all units have
  been activated. Each unit is activated once.

  The `activate_ratio` is in (0, 1].
  """

  def get_mask(pre_mask: tf.Tensor, iter_step: int) -> tf.Tensor:
    mask_ratio = 1 - (1 - activate_ratio) ** iter_step
    mask = tf.where(random(pre_mask.shape, seed) < mask_ratio, 1., 0.)
    mask = tf.where(pre_mask > 0, pre_mask, mask)
    return mask

  ambient_mask = tf.zeros([bm.ambient_size])
  latent_mask = tf.zeros([bm.latent_size])
  iter_step = 1
  while True:
    ambient_mask = get_mask(ambient_mask, iter_step)
    latent_mask = get_mask(latent_mask, iter_step)
    state = deterministic_activate(bm, state, ambient_mask, latent_mask)
    iter_step += 1
    if tf.reduce_min(ambient_mask) * tf.reduce_min(latent_mask) > 0:
      # all have been activated
      break
  return state


def mean_field_approx(bm: BernoulliBoltzmannMachine,
                      ambient: tf.Tensor,
                      max_step: int,
                      tolerance: float,
                      seed: int):
  """Returns the final distribution and the final step.

  Step starts at one and ends, if not breaking up, at the `max_step`.
  """
  # abbreviations
  v = ambient
  W = bm.ambient_latent_kernel
  J = bm.latent_latent_kernel
  bh = bm.latent_bias

  batch_size = get_batch_size(v)
  mu = random([batch_size, bm.latent_size], seed)
  step = 1
  while step <= max_step:
    new_mu = tf.sigmoid(v @ W + mu @ J + bh)
    if infinity_norm(new_mu - mu) < tolerance:
      break
    mu = new_mu
    step += 1
  return Bernoulli(mu), step


def initialize_fantasy_state(bm: BernoulliBoltzmannMachine,
                             num_samples: int,
                             seed: int):
  latent_p = 0.5 * tf.ones([num_samples, bm.latent_size])
  latent = Bernoulli(latent_p).sample(seed)

  ambient_p = 0.5 * tf.ones([num_samples, bm.ambient_size])
  ambient = Bernoulli(ambient_p).sample(seed)
  ambient = bm.activate(State(ambient, latent)).ambient
  # XXX: why not directly return bm.activate(State(ambient, latent))
  return State(ambient, latent)


class LogInternalInformation(Callback):

  def __init__(self,
               bm: BernoulliBoltzmannMachine,
               log_step: int,
               verbose: bool):
    self.bm = bm
    self.log_step = log_step
    self.verbose = verbose

    self.history = History()

  def __call__(self,
               step: int,
               real_ambient: tf.Tensor,
               _: State):
    if step % self.log_step != 0:
      return

    def stats(x, name):
      mean, var = tf.nn.moments(x, axes=range(len(x.shape)))
      std = tf.sqrt(var)
      self.history.log(step, f'{name}', f'{mean:.5f} ({std:.5f})')

    real_latent = (
        self.bm.get_latent_given_ambient(real_ambient)
        .sample(self.bm.seed))
    stats(real_latent, 'real_latent')
    for param, _ in self.bm.params_and_ops:
      stats(param, param.name)

    recon_error = get_reconstruction_error(self.bm, real_ambient)
    self.history.log(step, 'recon_error', recon_error)

    if self.verbose:
      print(self.history.show(step))


def get_reconstruction_error(bm: BernoulliBoltzmannMachine,
                             ambient: tf.Tensor):

  def norm(x: tf.Tensor) -> float:
    return tf.reduce_mean(tf.where(x != 0, 1., 0.))

  return B.get_reconstruction_error(bm, ambient, norm)


def is_restricted(bm: BernoulliBoltzmannMachine):
  return not bm.connect_ambient_to_ambient and not bm.connect_latent_to_latent


class LatentIncrementingInitializer(Initializer):

  def __init__(self, base_bm: BernoulliBoltzmannMachine, increment: int):
    self.base_bm = base_bm
    self.increment = increment

  @property
  def ambient_ambient_kernel(self):

    def initializer(*_):
      return self.base_bm.ambient_ambient_kernel

    return initializer

  @property
  def ambient_bias(self):

    def initializer(*_):
      return self.base_bm.ambient_bias

    return initializer

  @property
  def latent_latent_kernel(self):

    def initializer(*_):
      W = self.base_bm.latent_latent_kernel
      W = tf.concat(
          [
              W,
              tf.zeros([W.shape[0], self.increment]),
          ],
          axis=1)
      W = tf.concat(
          [
              W,
              tf.zeros([self.increment, W.shape[1]]),
          ],
          axis=0)
      return W

    return initializer

  @property
  def latent_bias(self):

    def initializer(*_):
      return tf.concat(
          [
              self.base_bm.latent_bias,
              tf.zeros([self.increment]),
          ],
          axis=0)

    return initializer

  @property
  def ambient_latent_kernel(self):

    def initializer(*_):
      return tf.concat(
          [
              self.base_bm.ambient_latent_kernel,
              tf.zeros([self.base_bm.ambient_size, self.increment]),
          ],
          axis=1)

    return initializer


def enlarge_latent(base_bm: BernoulliBoltzmannMachine,
                   base_fantasy_state: State,
                   increment: int):
  """Enlarges the latent size of BM `base_bm` by `increment`, and returns
  the enlarged BM and fantasy state.

  Suppose that the base BM and the base fantasy state have been trained.
  """
  config = base_bm.get_config()
  config['latent_size'] += increment
  config['initializer'] = LatentIncrementingInitializer(base_bm, increment)
  bm = BernoulliBoltzmannMachine(**config)

  fantasy_ambient = base_fantasy_state.ambient

  batch_size = get_batch_size(fantasy_ambient)
  p = 0.5 * tf.ones([batch_size, increment])
  inc_fantasy_latent = Bernoulli(p).sample(base_bm.seed)
  fantasy_latent = tf.concat(
      [
          base_fantasy_state.latent,
          inc_fantasy_latent,
      ],
      axis=1)
  fantasy_state = State(fantasy_ambient, fantasy_latent)
  return bm, fantasy_state
