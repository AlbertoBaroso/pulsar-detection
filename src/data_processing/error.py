from data_processing.utils import one_dimensional_array
import numpy


def error_rate(predicted: numpy.ndarray, expected: numpy.ndarray) -> float:
    """ 
        Compute the error rate of a prediction as (#correctly_predicted / #total_samples) 

        Parameters
        ----------
        predicted (numpy.ndarray): The predicted labels
        expected (numpy.ndarray): The expected labels
        
        Returns
        -------
        (float): The error rate
    """
    
    correct = sum(one_dimensional_array(predicted) == one_dimensional_array(expected))
    accuracy = correct / expected.shape[1]
    error = 1 - accuracy
    return error / float(len(predicted))