from data_processing.analytics import covariance_matrix, empirical_mean
from data_processing.utils import vrow, vcol
import numpy
import scipy


class MVG:
    @staticmethod
    def logpdf_GAU_ND(X: numpy.ndarray, mu: numpy.ndarray, C: numpy.ndarray) -> numpy.ndarray:
        """
        Compute the log-density of a multivariate Gaussian distribution for all samples

        Args:
            X (numpy.ndarray):  Original dataset, of shape (n, m) where n is the number of features and m is the number of samples
            mu (numpy.ndarray): Mean of the MVG distribution, it has shape (n, 1)
            C (numpy.ndarray):  Covariance matrix of the MVG distribution, it has shape (n, n)

        Returns:
            numpy.ndarray: the log-density of the MVG distribution computed for each sample, it has shape (m, 1)
        """
        M = X.shape[0]
        P = numpy.linalg.inv(C)  # Covariance matrix inverse
        determinant_logarithm = numpy.linalg.slogdet(P)[1]
        centered_data = X - mu

        return -0.5 * M * numpy.log(2 * numpy.pi) + 0.5 * determinant_logarithm - 0.5 * (numpy.dot(P, centered_data) * centered_data).sum(0)

    @staticmethod
    def train_multivariate_gaussian(
        DTR: numpy.ndarray, LTR: numpy.ndarray, labels: numpy.ndarray
    ) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """
        Compute the empirical mean and covariance matrix of each class

        Args:
            DTR (numpy.ndarray):     Training dataset, of shape (n, m) where n is the number of features and m is the number of samples
            LTR (numpy.ndarray):     Training labels, of shape (m, 1)
            labels (numpy.ndarray):  List of labels [0, 1, ..., k-1], where k is the number of classes

        Returns:
            µ (list(numpy.ndarray)): List of empirical means of each class, each element has shape (n, 1)
            Σ (list(numpy.ndarray)): List of covariance matrices of each class, each element has shape (n, n)
        """

        # Estimate model parameters
        µ, Σ = [], []

        for label in labels:
            # Compute Maximum likelihood estimate of µ and Σ for each class
            samples = DTR[:, LTR == label]
            µ.append(empirical_mean(samples))
            Σ.append(covariance_matrix(samples))

        return µ, Σ

    @staticmethod
    def train_naive_bayes_mvg(
        DTR: numpy.ndarray, LTR: numpy.ndarray, labels: numpy.ndarray
    ) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """
        Compute the empirical mean and diagonal covariance matrix of each class

        Args:
            DTR (numpy.ndarray):     Training dataset, of shape (n, m) where n is the number of features and m is the number of samples
            LTR (numpy.ndarray):     Training labels, of shape (m, 1)
            labels (numpy.ndarray):  List of labels [0, 1, ..., k-1], where k is the number of classes

        Returns:
            µ (list(numpy.ndarray)): List of empirical means of each class, each element has shape (n, 1)
            Σ (list(numpy.ndarray)): List of diagonal covariance matrices of each class, each element has shape (n, n)
        """

        # Estimate model parameters
        µ, Σ = [], []

        for label in labels:
            # Compute Maximum likelihood estimate of µ and Σ for each class
            samples = DTR[:, LTR == label]
            µ.append(empirical_mean(samples))
            # Keep only the values on the diagonal by multiplying the covariance matrix by the identity matrix
            Σ_c = covariance_matrix(samples) * numpy.identity(samples.shape[0])
            Σ.append(Σ_c)

        return µ, Σ

    @staticmethod
    def train_tied_covariance_mvg(
        DTR: numpy.ndarray, LTR: numpy.ndarray, labels: numpy.ndarray
    ) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """
        Compute the empirical mean of each class and the tied covariance matrix

        Args:
            DTR (numpy.ndarray):     Training dataset, of shape (n, m) where n is the number of features and m is the number of samples
            LTR (numpy.ndarray):     Training labels, of shape (m, 1)
            labels (numpy.ndarray):  List of labels [0, 1, ..., k-1], where k is the number of classes

        Returns:
            µ (list(numpy.ndarray)): List of empirical means of each class, each element has shape (n, 1)
            Σ (numpy.ndarray):       Tied Covariance matrix with shape (n, n)
        """

        # Estimate model parameters
        µ, Σ = [], numpy.zeros((DTR.shape[0], DTR.shape[0]))

        for label in labels:
            # Compute Maximum likelihood estimate of µ and Σ for each class
            samples = DTR[:, LTR == label]
            µ.append(empirical_mean(samples))
            # Weighted sum (by number of samples) of all the covariance matrices
            Σ += covariance_matrix(samples) * float(samples.shape[1])

        # Average over the total number of samples
        Σ /= DTR.shape[1]

        return µ, Σ

    @staticmethod
    def train_tied_covariance_naive_bayes_mvg(
        DTR: numpy.ndarray, LTR: numpy.ndarray, labels: numpy.ndarray
    ) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """
        Compute the empirical mean of each class and the diagonal tied covariance matrix

        Args:
            DTR (numpy.ndarray):     Training dataset, of shape (n, m) where n is the number of features and m is the number of samples
            LTR (numpy.ndarray):     Training labels, of shape (m, 1)
            labels (numpy.ndarray):  List of labels [0, 1, ..., k-1], where k is the number of classes

        Returns:
            µ (list(numpy.ndarray)): List of empirical means of each class, each element has shape (n, 1)
            Σ (numpy.ndarray):       Tied Diagonal Covariance matrix with shape (n, n)
        """

        # Estimate model parameters
        µ, Σ = [], numpy.zeros((DTR.shape[0], DTR.shape[0]))

        for label in labels:
            # Compute Maximum likelihood estimate of µ and Σ for each class
            samples = DTR[:, LTR == label]
            µ.append(empirical_mean(samples))
            # Weighted sum (by number of samples) of all the covariance matrices, keep only the values on the diagonal
            Σ += covariance_matrix(samples) * numpy.identity(samples.shape[0]) * float(samples.shape[1])

        # Average over the total number of samples
        Σ /= DTR.shape[1]

        return µ, Σ

    @staticmethod
    def score_samples(
        DTE: numpy.ndarray, µ: list[numpy.ndarray], Σ: list[numpy.ndarray], priors: numpy.ndarray, labels: numpy.ndarray, tied: bool
    ) -> numpy.ndarray:
        """
        Compute the posterior probability for each class,
        Assign to each sample the class with the highest posterior probability

        Args:
            DTE (numpy.ndarray):        Test dataset, of shape (n, m) where n is the number of features and m is the number of samples
            µ (list[numpy.ndarray]):    List of empirical means of each class, each element has shape (n, 1)
            Σ (list[numpy.ndarray]):    Covariance matrix with shape (n, n) if tied is True, otherwise a list of covariance matrices of each class
            priors (numpy.ndarray):     List of prior probability of each class
            labels (numpy.ndarray):     List of labels [0, 1, ..., k-1], where k is the number of classes
            tied (bool):                True if the covariance matrix is tied, False otherwise

        Returns:
            numpy.ndarray:              List of logarithms of the joint distribution for samples and classes
        """

        # Compute class-conditional probabilities for each class
        S = [MVG.logpdf_GAU_ND(DTE, µ[c], Σ if tied else Σ[c]) for c in range(len(labels))]

        # Combine the scores with prior information
        log_priors = vcol(numpy.log(priors))
        log_SJoint = S + log_priors

        return log_SJoint

    @staticmethod
    def predict_samples(log_SJoint: numpy.ndarray) -> numpy.ndarray:
        """
        Assign to each sample the class with the highest posterior probability

        Args:
            log_SJoint (numpy.ndarray): List of logarithms of the joint distribution for samples and classes

        Returns:
           (numpy.ndarray):             Predicted labels for each sample, it has shape (m, 1)
        """

        # Compute Marginal probability with log-sum-exp trick
        log_SMarginal = vrow(scipy.special.logsumexp(log_SJoint, axis=0))

        # Compute posterior probabilities
        log_SPost = log_SJoint - log_SMarginal
        SPost = numpy.exp(log_SPost)  # Posterior = exponent of the log-posterior

        # Select the class with the highest posterior probability
        return SPost.argmax(0)
