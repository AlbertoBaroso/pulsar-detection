from data_processing.analytics import covariance_matrix
from data_processing.utils import vcol
import scipy
import numpy

def most_discriminant_egienvectors(C: numpy.ndarray, m: int) -> numpy.ndarray:
    """
        Compute the most discriminant eigenvectors given a covariance matrix C
        
        Parameters
        ----------            
        C (numpy.ndarray): covariance matrix
        m (int): number of dimensions to reduce to
        
        Returns
        -------
        The m most discriminant eigenvectors of C
    """

    _s, U = numpy.linalg.eigh(C)
    # _s: eigenvalues sorted from smallest to largest
    # U: eigenvectors (columns of U)
    P = U[:, ::-1][:, 0:m] # Retrieve only the m principal directions

    return P
    

def lda(D, L, m):

    pulsar = D[:, L == 0]
    non_pulsar = D[:, L == 1]

    # Compute covariance matrices
    C_pulsar = covariance_matrix(pulsar)
    C_non_pulsar = covariance_matrix(non_pulsar)

    n_samples_pulsar = pulsar.shape[1]
    n_samples_non_pulsar = non_pulsar.shape[1]
    N = n_samples_pulsar + n_samples_non_pulsar
    SW = ((C_pulsar * n_samples_pulsar) +
          (C_non_pulsar * n_samples_non_pulsar)) / N

    # mu is the global mean of the dataset
    mu = D.mean(1)
    mu_pulsar = vcol(pulsar.mean(axis=1) - mu)
    mu_non_pulsar = vcol(non_pulsar.mean(axis=1) - mu)
    SB = ((n_samples_pulsar * numpy.dot(mu_pulsar, mu_pulsar.T)) +
          (n_samples_non_pulsar * numpy.dot(mu_non_pulsar, mu_non_pulsar.T))) / N

    ### SOLVE THE GENERALIZED EIGENVALUE PROBLEM ###
    # m: at most n_of_classes - 1 discriminant directions

    # Be careful, not numpy.linalg.eigh, because the numpy won't solve the generalized eigenvalue problem
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]

    # Solving the eigenvalue problem by joint diagonalization of SB and SW

    # The first step consists in estimating matrix P1 such that the within class covariance of the transformed points P1x is the identity.

    U, s, _ = numpy.linalg.svd(SW)
    # s is a 1-dimensional array containing the diagonal of Σ

    # UΣU.T is the SVD of SW
    # P 1 = UΣ^(−1/2)U.T
    P1 = numpy.dot(numpy.dot(U, numpy.diag(1.0/(s**0.5))), U.T)
    # numpy.diag builds a diagonal matrix from a one-dimensional array
    # (1.0/(s**0.5) = The diagonal of Σ^(-1/2)

    # The transformed between class covariance SBT can be computed as
    SBT = numpy.dot(numpy.dot(P1, SB), P1.T)

    # Compute the matrix P2 composed of the m most discriminant eigenvectors of SBT
    P2 = most_discriminant_egienvectors(SBT, m)

    # Transform from the original space to the LDA subspace
    W = numpy.dot(P1.T, P2)  # LDA matrix
    # y = numpy.dot(P2.T, P1, x)
    Y = numpy.dot(W.T, D)  # the solution is not orthogonal

    # Scatterplot
    # pulsar = Y[:, L == 1]
    # non_pulsar = Y[:, L == 0]
    # plt.plot(pulsar[0], linestyle='', marker='.', markersize=10, label="Pulsar")
    # plt.plot(non_pulsar[0], linestyle='', marker='.', markersize=10, label="Non pulsar")
    # plt.plot()
    # plt.legend()
    # plt.show()

    return Y


def pca(D: numpy.ndarray, m: int) -> numpy.ndarray:
    """ 
        Principal Component Analysis (PCA) 

        Parameters
        ----------
        D (numpy.ndarray): data matrix
        m (int): number of dimensions to reduce to
    
        Returns
        -------
        (numpy.ndarray): the m directions over which data has most variance
    """

    # Compute means for each column of the matrix
    mu = D.mean(axis=1)

    # Compute the 0-mean matrix (centered data)
    DC = D - vcol(mu)

    # Compute covariance matrix
    C = numpy.dot(DC, DC.T) / float(D.shape[1])

    # Retrieve The m leading eigenvectors from U (Principal components)
    # (We reverse the order of the columns of U so that the leading eigenvectors are in the first m columns):
    directions = most_discriminant_egienvectors(C, m)

    projected_data = numpy.dot(directions.T, D)

    # import matplotlib.pyplot as plt
    # pulsar = projected_data[:, labels == 1]
    # non_pulsar = projected_data[:, labels == 0]
    # plt.plot(pulsar[0], pulsar[1], linestyle='', marker='.', markersize=10, label="Pulsar")
    # plt.plot(non_pulsar[0], non_pulsar[1], linestyle='', marker='.', markersize=10, label="Non pulsar")
    # plt.plot()
    # plt.legend()
    # plt.show()

    return projected_data
