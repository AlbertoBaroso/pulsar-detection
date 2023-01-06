from data_processing.analytics import confusion_matrix, fnr_fpr
from data_processing.utils import one_dimensional_array
import numpy


def error_rate(predicted: numpy.ndarray, expected: numpy.ndarray) -> float:
    """
    Compute the error rate of a prediction as (#correctly_predicted / #total_samples)

    Args:
        predicted (numpy.ndarray): The predicted labels
        expected (numpy.ndarray): The expected labels

    Returns:
        (float): The error rate
    """
    expected_1d = one_dimensional_array(expected)
    correct = sum(one_dimensional_array(predicted) == expected_1d)
    accuracy = correct / len(expected_1d)
    error = 1 - accuracy
    return error

def optimal_bayes_decisions(scores: numpy.ndarray, π1: float, Cfn: float, Cfp: float) -> numpy.ndarray:
    """
    Compute the optimal Bayes decision given the scores, class priors and cost matrix

    Args:
        scores (numpy.ndarray): Scores of the test samples
        Cfn (float): Cost of false negative predictions
        Cfp (float): Cost of false positive predictions

    Returns:
        numpy.ndarray: Optimal Bayes decisions
    """
    # C = numpy.array([[0, Cfn], [Cfp, 0]]) # Cost matrix
    # π = numpy.array([1 - π1, π1]) # Class priors
    t = -numpy.log(π1 * Cfn / ((1 - π1) * Cfp))  # Threshold
    return (scores > t).astype(int)

def optimal_bayes_decisions_threshold(scores: numpy.ndarray, threshold: float) -> numpy.ndarray:
    """
    Compute the optimal Bayes decision given the scores and a threshold

    Args:
        scores (numpy.ndarray): Scores of the test samples
        threshold (float): Threshold for the Bayes decision

    Returns:
        numpy.ndarray: Optimal Bayes decisions
    """
    return numpy.int32(scores > threshold)

def unnormalized_DCF(confusion_matrix: numpy.ndarray, π1: float, Cfn: float, Cfp: float) -> float:
    """
    Un-normalized Detection cost function === Empirical Bayes risk

    Args:
        confusion_matrix (numpy.ndarray): Array of shape (K, K) where K is the number of classes
        π1  (float): Prior probability of the first class
        Cfn (float): Cost of false negative
        Cfp (float): Cost of false positive

    Returns:
        float: Un-normalized Detection cost function
    """
    FNR, FPR = fnr_fpr(confusion_matrix)
    return π1 * Cfn * FNR + (1 - π1) * Cfp * FPR


def normalized_DCF(confusion_matrix: numpy.ndarray, π1: float, Cfn: float, Cfp: float) -> float:
    """
    Normalize the Detection cost function using the best of possible dummy systems

    Args:
        confusion_matrix (numpy.ndarray): Array of shape (K, K) where K is the number of classes
        Cfn (float): Cost of false negative predictions
        Cfp (float): Cost of false positive predictions

    Returns:
        float: Normalized Detection cost function
    """
    return unnormalized_DCF(confusion_matrix, π1, Cfn, Cfp) / min(π1 * Cfn, (1 - π1) * Cfp)


def minimum_DCF(scores: numpy.ndarray, labels: numpy.ndarray, π1: float, Cfn: float, Cfp: float) -> float:
    """
    Minimum Detection cost function

    Args:
        scores (numpy.ndarray): Scores assigned to the test samples
        labels (numpy.ndarray): Labels of the test samples
        π1 (float):             Prior probability of the first class
        Cfn (float):            Cost of false negative predictions
        Cfp (float):            Cost of false positive predictions

    Returns:
        float: Minimum Detection cost function
    """
    
    thresholds = [-numpy.inf] + sorted(scores) + [numpy.inf]
    min_DCF = numpy.inf
    for threshold in thresholds:
        predictions = optimal_bayes_decisions_threshold(scores, threshold)
        CM = confusion_matrix(predictions, labels)
        min_DCF = min(min_DCF, normalized_DCF(CM, π1, Cfn, Cfp))
    return min_DCF
