from scipy.stats import norm
from typing import Optional
import numpy


### GAUSSIANIZATION ###

import math

def rank_samples(feature: numpy.ndarray, training_samples: Optional[numpy.ndarray]) -> list[float]:
    """
    Compute, for each value, the percentage of values smaller than the current value

    Args:
        feature (numpy.ndarray): feature to rank
        training_samples (Optional[numpy.ndarray]): training samples to rank against
        
    Returns:
        list[float]: list of ranks
    """

    comparison_feature = feature if training_samples is None else training_samples
    ranks = []
    for x in feature:
        less_than_x = 0
        for x_i in comparison_feature:  # x_i is the value of the considered feature for the i-th training sample
            if x_i < x:
                less_than_x += 1
        rank = (less_than_x + 1) / (comparison_feature.shape[0] + 2)
        # +2 because we assume the existance of a feature smaller than all the others and a feature larger than all the others)
        ranks.append(rank)
    return ranks


def rank_percentile(rank: list[float]) -> numpy.ndarray:
    """ Compute percentile (percent point function) of the rank """
    return norm.ppf(rank)


def gaussianize_feature(feature, training_samples=None) -> numpy.ndarray:
    """ Transform a feature into a gaussian distribution """
    return numpy.array(rank_percentile(rank_samples(feature, training_samples)))


def gaussianize_training_samples(DTR: numpy.ndarray) -> numpy.ndarray:
    """ Transform each feature of the training samples into a gaussian distribution """
    return numpy.array([gaussianize_feature(feature) for feature in DTR])


def gaussianize_test_samples(DTE: numpy.ndarray, DTR: numpy.ndarray) -> numpy.ndarray:
    """ Compute gaussianization by ranking test samples against training samples """
    result = []
    for i in range(len(DTE)):
        gaussianized_feature = gaussianize_feature(DTE[i], DTR[i])
        result.append(gaussianized_feature)
    return numpy.vstack(result)
