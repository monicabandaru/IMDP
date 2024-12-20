from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from differential_privacy.optimizer import dp_query
from tensorflow.python.util import nest
#nest = tf.compat.v1.contrib.framework.nest


class GaussianSumQuery(dp_query.DPQuery):
  """Implements DPQuery interface for Gaussian sum queries.

  Accumulates clipped vectors, then adds Gaussian noise to the sum.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ['l2_norm_clip', 'stddev'])

  def __init__(self, l2_norm_clip, stddev):
    """Initializes the GaussianSumQuery.

    Args:
      l2_norm_clip: The clipping norm to apply to the global norm of each
        record.
      stddev: The stddev of the noise added to the sum.
    """
    self._l2_norm_clip = l2_norm_clip
    self._stddev = stddev

  def initial_global_state(self):
    """Returns the initial global state for the GaussianSumQuery."""
    return self._GlobalState(float(self._l2_norm_clip), float(self._stddev))

  def derive_sample_params(self, global_state):
    """Given the global state, derives parameters to use for the next sample.

    Args:
      global_state: The current global state.

    Returns:
      Parameters to use to process records in the next sample.
    """
    return global_state.l2_norm_clip

  def initial_sample_state(self, global_state, tensors):
    """Returns an initial state to use for the next sample.

    Args:
      global_state: The current global state.
      tensors: A structure of tensors used as a template to create the initial
        sample state.

    Returns: An initial sample state.
    """
    del global_state  # unused.
    return nest.map_structure(tf.zeros_like, tensors)

  def accumulate_record(self, params, sample_state, record):
    """Accumulates a single record into the sample state.

    Args:
      params: The parameters for the sample.
      sample_state: The current sample state.
      record: The record to accumulate.

    Returns:
      The updated sample state.
    """
    l2_norm_clip = params
    record_as_list = nest.flatten(record)
    clipped_as_list, _ = tf.clip_by_global_norm(record_as_list, l2_norm_clip)
    clipped = nest.pack_sequence_as(record, clipped_as_list)
    return nest.map_structure(tf.add, sample_state, clipped)

  def get_noised_result(self, sample_state, global_state, noise_flag=True):
    #print("SUM Query get_noised_result")
    """Gets noised sum after all records of sample have been accumulated.

    Args:
      sample_state: The sample state after all records have been accumulated.
      global_state: The global state.

    Returns:
      A tuple (estimate, new_global_state) where "estimate" is the estimated
      sum of the records and "new_global_state" is the updated global state.
    """
    def add_noise(v):
      if noise_flag:
        return v + tf.compat.v1.random_normal(tf.shape(v), stddev=global_state.stddev)
      else:
        return v


    return nest.map_structure(add_noise, sample_state), global_state


class GaussianAverageQuery(dp_query.DPQuery):
  """Implements DPQuery interface for Gaussian average queries.

  Accumulates clipped vectors, adds Gaussian noise, and normalizes.

  Note that we use "fixed-denominator" estimation: the denominator should be
  specified as the expected number of records per sample. Accumulating the
  denominator separately would also be possible but would be produce a higher
  variance estimator.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple(
      '_GlobalState', ['sum_state', 'denominator'])

  def __init__(self, l2_norm_clip, sum_stddev, denominator):
    """Initializes the GaussianAverageQuery.

    Args:
      l2_norm_clip: The clipping norm to apply to the global norm of each
        record.
      sum_stddev: The stddev of the noise added to the sum (before
        normalization).
      denominator: The normalization constant (applied after noise is added to
        the sum).
    """
    self._numerator = GaussianSumQuery(l2_norm_clip, sum_stddev)
    self._denominator = denominator

  def initial_global_state(self):
    """Returns the initial global state for the GaussianAverageQuery."""
    sum_global_state = self._numerator.initial_global_state()
    return self._GlobalState(sum_global_state, float(self._denominator))

  def derive_sample_params(self, global_state):
    """Given the global state, derives parameters to use for the next sample.

    Args:
      global_state: The current global state.

    Returns:
      Parameters to use to process records in the next sample.
    """
    return self._numerator.derive_sample_params(global_state.sum_state)

  def initial_sample_state(self, global_state, tensors):
    """Returns an initial state to use for the next sample.

    Args:
      global_state: The current global state.
      tensors: A structure of tensors used as a template to create the initial
        sample state.

    Returns: An initial sample state.
    """
    # GaussianAverageQuery has no state beyond the sum state.
    return self._numerator.initial_sample_state(global_state.sum_state, tensors)

  def accumulate_record(self, params, sample_state, record):
    """Accumulates a single record into the sample state.

    Args:
      params: The parameters for the sample.
      sample_state: The current sample state.
      record: The record to accumulate.

    Returns:
      The updated sample state.
    """
    return self._numerator.accumulate_record(params, sample_state, record)

  def get_noised_result(self, sample_state, global_state, add_noise=True):
    #print("Average Query get_noised_result")
    """Gets noised average after all records of sample have been accumulated.

    Args:
      sample_state: The sample state after all records have been accumulated.
      global_state: The global state.

    Returns:
      A tuple (estimate, new_global_state) where "estimate" is the estimated
      average of the records and "new_global_state" is the updated global state.
    """
    noised_sum, new_sum_global_state = self._numerator.get_noised_result(
        sample_state, global_state.sum_state, add_noise)
    new_global_state = self._GlobalState(
        new_sum_global_state, global_state.denominator)
    def normalize(v):
      return tf.truediv(v, global_state.denominator)

    return nest.map_structure(normalize, noised_sum), new_global_state