import numpy
import sys
sys.path.append("./")
from data_processing.utils import vcol


def load_features():
    return [
        "Mean of the integrated profile",
        "Standard deviation of the integrated profile",
        "Excess kurtosis of the integrated profile",
        "Skewness of the integrated profile",
        "Mean of the DM-SNR curve", 
        "Standard deviation of the DM-SNR curve",
        "Excess kurtosis of the DM-SNR curve",
        "Skewness of the DM-SNR curve"
    ]


def load_data(dataset):
    samples = []
    labels = []

    # Read samples
    with open(dataset, 'r') as file:
        try:
            for line in file:

                attributes = line.split(",")

                # Extract first 8 attributes as floats
                sample = vcol(numpy.array([float(attr) for attr in attributes[0:8]]))

                # Retrieve class label
                label = int(attributes[-1].strip())

                samples.append(sample)
                labels.append(label)
        except:
            pass

    return numpy.hstack(samples), numpy.array(labels)

# Retrieve training samples
def load_training():
    return load_data('../data/Train.txt')

# Retrieve test samples
def load_test():
    return load_data('../data/Test.txt')
