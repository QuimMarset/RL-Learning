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

def print_model_to_json_file(model, file_path):
    with open(file_path, "w") as write_file:
        json_arguments = {'indent' : 4, 'separators' : (', ', ': ')}
        json_string = model.to_json(**json_arguments)
        write_file.write(json_string)