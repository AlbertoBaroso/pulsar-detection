from data_processing.utils import vcol
import numpy


def load_features() -> list[str]:
    """ Load the names of the features """
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


def load_data(dataset: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    """ Read the dataset and return the samples and labels """

    samples = []
    labels = []

    # Read samples
    with open(dataset, 'r') as file:
        try:
            for line in file:

                attributes = line.split(",")

                # Extract first 8 attributes as floats
                sample = vcol(numpy.array([float(attr)
                              for attr in attributes[0:8]]))

                # Retrieve class label
                label = int(attributes[-1].strip())

                samples.append(sample)
                labels.append(label)
        except:
            pass

    return numpy.hstack(samples), numpy.array(labels)


def load_training() -> tuple[numpy.ndarray, numpy.ndarray]:
    """ Retrieve training samples """
    return load_data('../data/Train.txt')


def load_test() -> tuple[numpy.ndarray, numpy.ndarray]:
    """ Retrieve test samples """
    return load_data('../data/Test.txt')
