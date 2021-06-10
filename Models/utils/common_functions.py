import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Categorical, MultivariateNormalDiag


def compute_log_of_tensor(values):
    offsets = tf.cast(values == 0, dtype = tf.float32)*1e-6
    values = values + offsets
    return tf.math.log(values)

def select_values_of_2D_tensor(tensor, second_dimension_indices):
    first_dimension = tensor.shape[0]
    first_dimension_indices = tf.constant(range(first_dimension), shape = (first_dimension, 1))
    indices = tf.concat([first_dimension_indices, tf.expand_dims(second_dimension_indices, axis = -1)], axis = -1)
    selected_values = tf.gather_nd(tensor, indices)
    return selected_values

def compute_pdf_of_gaussian_samples(mus, log_sigmas, samples):
    sigmas = tf.exp(log_sigmas)
    normal_distributions = MultivariateNormalDiag(mus, sigmas)
    return normal_distributions.prob(samples)

def sample_from_gaussians(mus, log_sigmas):
    sigmas = tf.exp(log_sigmas)
    normal_distribs = MultivariateNormalDiag(mus, sigmas)
    return normal_distribs.sample()

def sample_from_categoricals(probability_distributions):
    categorical_distribs = Categorical(probs = probability_distributions)
    return categorical_distribs.sample()

def sample_from_bounded_gaussian(mus, log_sigmas):
    unbounded_samples = sample_from_gaussians(mus, log_sigmas)
    samples = tf.tanh(unbounded_samples)
    return samples, unbounded_samples

def compute_pdf_of_bounded_gaussian_samples(mus, log_sigmas, unbounded_samples):
    unbounded_samples_prob = compute_pdf_of_gaussian_samples(mus, log_sigmas, unbounded_samples)
    unbounded_samples_log_prob = compute_log_of_tensor(unbounded_samples_prob)
    log_jacobian_determinant = tf.reduce_sum(tf.math.log(1 - tf.tanh(unbounded_samples)**2 + 1e-6), axis = -1)
    samples_log_prob = unbounded_samples_log_prob - log_jacobian_determinant
    return samples_log_prob

def compute_kl_divergence_of_gaussians(mus, log_sigmas, old_mus, old_log_sigmas):
    sigmas = tf.exp(log_sigmas)
    old_sigmas = tf.exp(old_log_sigmas)
    normal_distribs = MultivariateNormalDiag(mus, sigmas)
    old_normal_distribs = MultivariateNormalDiag(old_mus, old_sigmas)
    kl_divergences = old_normal_distribs.kl_divergence(normal_distribs)
    return kl_divergences

def compute_kl_divergence_of_categorical(old_prob_dists, prob_dists):
    log_old_probs_dists = compute_log_of_tensor(old_prob_dists)
    log_probs_dists = compute_log_of_tensor(prob_dists)
    kl_divergences = tf.reduce_sum(old_prob_dists*(log_old_probs_dists - log_probs_dists), axis = -1)
    return kl_divergences

