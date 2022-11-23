from data_processing.analytics import covariance_matrix, between_class_covariance_matrix, within_class_covariance_matrix
import scipy
import numpy

def most_discriminant_eigenvectors(C: numpy.ndarray, m: int) -> numpy.ndarray:
    """
    Compute the most discriminant eigenvectors given a covariance matrix C

    Args:
        C (numpy.ndarray): covariance matrix
        m (int): number of dimensions to reduce to

    Returns:
        The m most discriminant eigenvectors of C
    """

    _s, U = numpy.linalg.eigh(C)
    # _s: eigenvalues sorted from smallest to largest
    # U: eigenvectors (columns of U)
    P = U[:, ::-1][:, 0:m]  # Retrieve only the m principal directions

    return P


def lda(D: numpy.ndarray, L: numpy.ndarray, m: int) -> numpy.ndarray:
    """
    Linear Discriminant Analysis (LDA)

    Args:
        D (numpy.ndarray): Data matrix
        L (numpy.ndarray): Labels of the samples
        m (int): Number of dimensions to reduce to, at most (# of classes - 1)

    Returns:
        numpy.ndarray: Samples projected onto the m most discriminant directions
    """

    # Get within and between class covariance matrices
    classes = sorted(set(L))
    SW = within_class_covariance_matrix(D, L, classes)
    SB = between_class_covariance_matrix(D, L, classes)

    # Solve the generalized eigenvalue problem
    _s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]  # LDA Projection matrix
    projected_data = numpy.dot(W.T, D)  # Project data

    return projected_data


def pca(D: numpy.ndarray, m: int) -> numpy.ndarray:
    """
    Principal Component Analysis (PCA)

    Args:
        D (numpy.ndarray): Data matrix
        m (int): Number of dimensions to reduce to

    Returns:
        (numpy.ndarray): Data projected onto the m directions with the largest variance
    """

    # Compute covariance matrix
    C = covariance_matrix(D)

    # Retrieve The m leading eigenvectors from U (Principal components)
    # (We reverse the order of the columns of U so that the leading eigenvectors are in the first m columns):
    directions = most_discriminant_eigenvectors(C, m)

    projected_data = numpy.dot(directions.T, D)  # Project data onto the m principal directions

    return projected_data
