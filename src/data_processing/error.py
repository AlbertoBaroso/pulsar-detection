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
