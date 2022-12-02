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