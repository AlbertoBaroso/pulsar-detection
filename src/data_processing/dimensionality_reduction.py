import numpy


def lda(D, L, m):

    from utils.analytics import covariance_matrix

    def most_discriminant_egienvectors(C, m):
        # Compute eigenvectors
        s, U = numpy.linalg.eigh(C)
        # s: eigenvalues sorted from smallest to largest
        # U: eigenvectors (columns of U)

        P = U[:, ::-1][:, 0:m]

        # Since the covariance matrix is semi-definite positive,
        # We can also get the sorted eigenvectors from the Singular Value Decomposition
        U2, s2, Vh = numpy.linalg.svd(C)
        # In this case, the singular values (which are equal to the eigenvalues) are sorted in descending order,
        # the columns of U are the corresponding eigenvectors
        P2 = U[:, 0:m]
        return P


    pulsar = D[:, L == 0]
    non_pulsar = D[:, L == 1]

    # Compute covariance matrix
    C_setosa = covariance_matrix(pulsar)
    C_versicolor = covariance_matrix(non_pulsar)

    n_samples_setosa = pulsar.shape[1]
    n_samples_versicolor = non_pulsar.shape[1]
    N = n_samples_setosa + n_samples_versicolor
    SW = ((C_setosa * n_samples_setosa) + (C_versicolor * n_samples_versicolor)) / N

    # mu is the global mean of the dataset
    mu = D.mean(1)
    mu_setosa = vcol(pulsar.mean(axis=1) - mu)
    mu_versicolor = vcol(non_pulsar.mean(axis=1) - mu)
    SB = ((n_samples_setosa * numpy.dot(mu_setosa, mu_setosa.T)) +
          (n_samples_versicolor * numpy.dot(mu_versicolor, mu_versicolor.T))) / N


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
    Y = numpy.dot(W.T, D) # the solution is not orthogonal

    # Scatterplot
    # pulsar = Y[:, L == 1]
    # non_pulsar = Y[:, L == 0]
    # plt.plot(pulsar[0], linestyle='', marker='.', markersize=10, label="Pulsar")
    # plt.plot(non_pulsar[0], linestyle='', marker='.', markersize=10, label="Non pulsar")
    # plt.plot()
    # plt.legend()
    # plt.show()

    return Y


def pca(D, m, labels):

    # @param C: covariance matrix
    # @param m: number of dimensions to reduce to
    def pca_projection_matrix(C, m):
        # Compute eigenvectors
        s, U = numpy.linalg.eigh(C)
        # s: eigenvalues sorted from smallest to largest
        # U: eigenvectors (columns of U)

        # Reverse the order of eigenvectors (Numpy returns them in ascending order)
        P = U[:, ::-1][:, 0:m] # Retrieve only the m principal directions
        return P


    # Compute means for each column of the matrix
    mu = D.mean(axis=1)

    # Compute the 0-mean matrix (centered data)
    DC = D - vcol(mu)

    # Compute covariance matrix
    C = numpy.dot(DC, DC.T) / float(D.shape[1])
    from utils.analytics import covariance_matrix
    C2 = covariance_matrix(DC)

    # Retrieve The m leading eigenvectors from U (Principal components)
    # (We reverse the order of the columns of U so that the leading eigenvectors are in the first m columns):
    directions = pca_projection_matrix(C, m)

    # Scatterplot
    # pulsar = D[:, labels == 1]
    # non_pulsar = D[:, labels == 0]
    # pulsar = numpy.dot(directions.T, pulsar)
    # non_pulsar = numpy.dot(directions.T, non_pulsar)
    # plt.plot(pulsar[0], pulsar[1], linestyle='', marker='.', markersize=10, label="Pulsar")
    # plt.plot(non_pulsar[0], non_pulsar[1], linestyle='', marker='.', markersize=10, label="Non pulsar")
    # plt.plot()
    # plt.legend()
    # plt.show()

    return numpy.dot(directions.T, D)