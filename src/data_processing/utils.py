import numpy


def vcol(array: numpy.ndarray) -> numpy.ndarray:
    """Reshape an array to a column array of shape (n, 1) where n is the number of elements in the array"""
    return array.reshape((array.size, 1))


def vrow(array: numpy.ndarray) -> numpy.ndarray:
    """Reshape an array to a row array of shape (1, n) where n is the number of elements in the array"""
    return array.reshape((1, array.size))


def one_dimensional_array(array: numpy.ndarray) -> numpy.ndarray:
    """Flatten a multi-dimensional array to a one-dimensional array"""
    return array.reshape((array.size,))


def project_data(D: numpy.ndarray, directions: numpy.ndarray) -> numpy.ndarray:
    """
    Project data using a projection matrix

    Args:
        D (numpy.ndarray):          Data to project
        directions (numpy.ndarray): Projection matrix

    Returns:
        numpy.ndarray: Samples projected in the space spanned by the directions
    """
    return numpy.dot(directions.T, D)  # Project data

def extended_data_matrix(D: numpy.ndarray, K:float=1.0):
    """
    Extend data matrix with a constant feature

    Args:
        D (numpy.ndarray):   Data matrix
        K (float, optional): Constant value to add to the data matrix. Defaults to 1.0.

    Returns:
        (numpy.ndarray): Extended data matrix
    """
    return numpy.vstack((D, numpy.full((1, D.shape[1]), K)))


def shuffle_data(D: numpy.ndarray, L: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Shuffle data and labels

    Args:
        D (numpy.ndarray): Data matrix
        L (numpy.ndarray): Label vector

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Shuffled data and labels
    """
    indices = numpy.arange(D.shape[1])
    numpy.random.seed(1)
    numpy.random.shuffle(indices)
    return D[:, indices], L[indices]