import numpy

def vcol(array: numpy.ndarray) -> numpy.ndarray:
    """ Reshape an array to a column array of shape (n, 1) where n is the number of elements in the array """
    return array.reshape((array.size, 1))


def vrow(array: numpy.ndarray) -> numpy.ndarray:
    """ Reshape an array to a row array of shape (1, n) where n is the number of elements in the array """
    return array.reshape((1, array.size))


def one_dimensional_array(array):
    return array.reshape((array.size,))




def split_80_20(samples, labels):
    split = int(samples.shape[1] * 0.8)
    training_samples = samples[:, :split]
    training_labels = labels[:split]
    test_samples = samples[:, split:]
    test_labels = labels[split:]
    return training_samples, training_labels, test_samples, test_labels