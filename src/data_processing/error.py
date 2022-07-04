import sys
sys.path.append("./")
from utils.preprocessing import one_dimensional_array


# Compute error rate as (#correctly_predicted / #total_samples)
def error_rate(predicted, expected):
    correct = sum(one_dimensional_array(predicted) == one_dimensional_array(expected))
    accuracy = correct / expected.shape[1]
    error = 1 - accuracy
    return error / float(len(predicted))