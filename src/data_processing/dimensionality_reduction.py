from data_processing.analytics import covariance_matrix, between_class_covariance_matrix, within_class_covariance_matrix
import scipy
import numpy


def pca(D: numpy.ndarray, m: int) -> numpy.ndarray:
    """
    Principal Component Analysis (PCA)

    Args:
        D (numpy.ndarray): Data matrix
        m (int): Number of dimensions to reduce to

    Returns:
        (numpy.ndarray): Array of the m directions with largest variance
    """

    # Compute covariance matrix
    C = covariance_matrix(D)

    # Retrieve The m leading eigenvectors from U (Principal components)
    _s, U = numpy.linalg.eigh(C)
    # _s: eigenvalues sorted from smallest to largest
    # U: eigenvectors (columns of U)

    # (Reverse the order of the columns of U so that the leading eigenvectors are in the first m columns):
    directions = U[:, ::-1][:, 0:m]  # Retrieve only the m principal directions

    return directions


def lda(D: numpy.ndarray, L: numpy.ndarray, m: int) -> numpy.ndarray:
    """
    Linear Discriminant Analysis (LDA)

    Args:
        D (numpy.ndarray): Data matrix
        L (numpy.ndarray): Labels of the samples
        m (int): Number of dimensions to reduce to, at most (# of classes - 1)

    Returns:
        numpy.ndarray: m most discriminant directions
    """

    # Get within and between class covariance matrices
    classes = sorted(set(L))
    SW = within_class_covariance_matrix(D, L, classes)
    SB = between_class_covariance_matrix(D, L, classes)

    # Solve the generalized eigenvalue problem
    _s, U = scipy.linalg.eigh(SB, SW)
    directions = U[:, ::-1][:, 0:m]  # LDA Projection matrix

    return directions
