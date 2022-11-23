from data_processing.data_load import load_training, load_features, load_test
from visualization.feature_plots import plot_feature_histogram
from scipy.stats import norm
from typing import Optional
import numpy


### GAUSSIANIZATION ###


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
        rank = (less_than_x + 1) / (feature.shape[0] + 2)
        # +2 because we assume the existance of a feature smaller than all the others and a feature larger than all the others)
        ranks.append(rank)
    return ranks


def rank_percentile(rank: list[float]) -> numpy.ndarray:
    """ Compute percentile (percent poin function) of the rank """
    return norm.ppf(rank)


def gaussianize_feature(feature, training_samples=None) -> numpy.ndarray:
    """ Transform a feature into a gaussian distribution """
    return numpy.array(rank_percentile(rank_samples(feature, training_samples)))


def gaussianize_training_samples(DTR: numpy.ndarray) -> numpy.ndarray:
    """ Transform each feature of the training samples into a gaussian distribution """
    return numpy.array([gaussianize_feature(x) for x in DTR])


def gaussianize_test_samples(DTE: numpy.ndarray, DTR: numpy.ndarray) -> numpy.ndarray:
    """ Compute gaussianization by raking test samples against training samples """
    result = []
    for i in range(len(DTE)):
        result.append(gaussianize_feature(DTE[i], DTR[i]))
    return numpy.vstack(result)


if __name__ == "__main__":

    features = load_features()
    training_samples, training_labels = load_training()
    test_samples, test_labels = load_test()

    # GAUSSIANIZATION OF SAMPLES #
    gaussianized_training_samples = gaussianize_training_samples(training_samples)
    gaussianized_test_samples = gaussianize_test_samples(test_samples, training_samples)

    for i in range(len(features)):
        plot_feature_histogram(features[i], gaussianized_training_samples[i], training_labels)
        plot_feature_histogram(features[i], gaussianized_test_samples[i], test_labels)
