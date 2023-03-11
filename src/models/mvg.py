from data_processing.analytics import covariance_matrix, empirical_mean
from data_processing.utils import vrow, vcol
from enum import Enum
import numpy
import scipy


# MVG Model types
class MVGModel(Enum):
    MVG = "Full Covariance"
    NAIVE = "Diagonal Covariance"
    TIED = "Tied Covariance"
    TIED_NAIVE = "Diagonal Tied Covariance"


class MVG:
    def __init__(self, type: MVGModel, DTR: numpy.ndarray, LTR: numpy.ndarray, labels: list[int]):
        """
        Train a multivariate gaussian model

        Args:
            type (MVGModel):         One of the MVGModel types: MVG, NAIVE, TIED, TIED_NAIVE
            DTR (numpy.ndarray):     Training dataset, of shape (n, m) where n is the number of features and m is the number of samples
            LTR (numpy.ndarray):     Training labels, of shape (m, 1)
            labels (numpy.ndarray):  List of labels [0, 1, ..., k-1], where k is the number of classes

        Raises:
            ValueError: Invalid MVGModel type
        """
        self.DTR = DTR
        self.LTR = LTR
        self.labels = labels
        self.type = type
        self.is_tied = type == MVGModel.TIED or type == MVGModel.TIED_NAIVE

        # Train the model
        if type == MVGModel.MVG:
            µ, Σ = self.train_multivariate_gaussian()
        elif type == MVGModel.NAIVE:
            µ, Σ = self.train_naive_bayes_mvg()
        elif type == MVGModel.TIED:
            µ, Σ = self.train_tied_covariance_mvg()
        elif type == MVGModel.TIED_NAIVE:
            µ, Σ = self.train_tied_covariance_naive_bayes_mvg()
        else:
            raise ValueError("Invalid MVG model type")

        self.µ, self.Σ = µ, Σ

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

    def train_multivariate_gaussian(self) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """
        Compute the empirical mean and covariance matrix of each class

        Returns:
            µ (list(numpy.ndarray)): List of empirical means of each class, each element has shape (n, 1)
            Σ (list(numpy.ndarray)): List of covariance matrices of each class, each element has shape (n, n)
        """

        # Estimate model parameters
        µ, Σ = [], []

        for label in self.labels:
            # Compute Maximum likelihood estimate of µ and Σ for each class
            samples = self.DTR[:, self.LTR == label]
            µ.append(empirical_mean(samples))
            Σ.append(covariance_matrix(samples))

        return µ, Σ

    def train_naive_bayes_mvg(self) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """
        Compute the empirical mean and diagonal covariance matrix of each class

        Returns:
            µ (list(numpy.ndarray)): List of empirical means of each class, each element has shape (n, 1)
            Σ (list(numpy.ndarray)): List of diagonal covariance matrices of each class, each element has shape (n, n)
        """

        # Estimate model parameters
        µ, Σ = [], []

        for label in self.labels:
            # Compute Maximum likelihood estimate of µ and Σ for each class
            samples = self.DTR[:, self.LTR == label]
            µ.append(empirical_mean(samples))
            # Keep only the values on the diagonal by multiplying the covariance matrix by the identity matrix
            Σ_c = covariance_matrix(samples) * numpy.identity(samples.shape[0])
            Σ.append(Σ_c)

        return µ, Σ

    def train_tied_covariance_mvg(self) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """
        Compute the empirical mean of each class and the tied covariance matrix

        Returns:
            µ (list(numpy.ndarray)): List of empirical means of each class, each element has shape (n, 1)
            Σ (numpy.ndarray):       Tied Covariance matrix with shape (n, n)
        """

        # Estimate model parameters
        µ, Σ = [], numpy.zeros((self.DTR.shape[0], self.DTR.shape[0]))

        for label in self.labels:
            # Compute Maximum likelihood estimate of µ and Σ for each class
            samples = self.DTR[:, self.LTR == label]
            µ.append(empirical_mean(samples))
            # Weighted sum (by number of samples) of all the covariance matrices
            Σ += covariance_matrix(samples) * float(samples.shape[1])

        # Average over the total number of samples
        Σ /= self.DTR.shape[1]

        return µ, Σ

    def train_tied_covariance_naive_bayes_mvg(self) -> tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """
        Compute the empirical mean of each class and the diagonal tied covariance matrix

        Returns:
            µ (list(numpy.ndarray)): List of empirical means of each class, each element has shape (n, 1)
            Σ (numpy.ndarray):       Tied Diagonal Covariance matrix with shape (n, n)
        """

        # Estimate model parameters
        µ, Σ = [], numpy.zeros((self.DTR.shape[0], self.DTR.shape[0]))

        for label in self.labels:
            # Compute Maximum likelihood estimate of µ and Σ for each class
            samples = self.DTR[:, self.LTR == label]
            µ.append(empirical_mean(samples))
            # Weighted sum (by number of samples) of all the covariance matrices, keep only the values on the diagonal
            Σ += covariance_matrix(samples) * numpy.identity(samples.shape[0]) * float(samples.shape[1])

        # Average over the total number of samples
        Σ /= self.DTR.shape[1]

        return µ, Σ

    def score_samples(self, DTE: numpy.ndarray, priors: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Compute the posterior probability for each class,
        Assign to each sample the class with the highest posterior probability

        Args:
            DTE (numpy.ndarray):     Test dataset, of shape (n, m) where n is the number of features and m is the number of samples
            priors (numpy.ndarray):  List of prior probability of each class

        Returns:
            numpy.ndarray:           List of logarithms of the joint distribution for samples and classes
            numpy.ndarray:           Log likelihood ratio for each sample
        """

        # Compute class-conditional probabilities for each class
        S = [MVG.logpdf_GAU_ND(DTE, self.µ[c], self.Σ if self.is_tied else self.Σ[c]) for c in range(len(self.labels))]

        # if α is not None and β is not None and π_tilde is not None:
        # for i in range(len(S)):
        #     S[i] = α * S[i] + β - numpy.log(π_tilde / (1 - π_tilde))

        # Compute Log Likelihood Ratios
        llr = S[1] - S[0]

        # Combine the scores with prior information
        log_priors = vcol(numpy.log(priors))
        log_SJoint = S + log_priors

        return log_SJoint, llr

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
