from data_processing.utils import vcol
import scipy.optimize
import numpy


# LOGISTIC REGRESSION #


class LogisticRegression:
    def __init__(self, DTR: numpy.ndarray, LTR: numpy.ndarray, λ: float, πT: float, quadratic: bool = False):
        """
        Args:
            DTR (numpy.ndarray): Training dataset
            LTR (numpy.ndarray): Labels for the training dataset
            λ (float):           Regularization parameter
            πT (float):          Prior probability of the first class
            quadratic (bool):    Whether to use a quadratic expanded feature space. Defaults to False.
        """
        
        if quadratic:
            DTR = self.features_expansion(DTR)
        
        Z = 2.0 * LTR - 1.0
        self.DTR_non_pulsar = DTR[:, LTR==0]
        self.DTR_pulsar = DTR[:, LTR==1]
        self.Z_non_pulsar = Z[LTR == 0]
        self.Z_pulsar = Z[LTR == 1]
        self.πT = πT
        self.λ = λ

        # Train the model
        x, _f, _d = scipy.optimize.fmin_l_bfgs_b(func=self.logreg_obj, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
        self.w, self.b = x[:-1], x[-1]

    def logreg_obj(self, v: numpy.ndarray):
        """
        Objective function for the logistic regression problem.

        Args:
            v (numpy.ndarray): Array with shape (D+1,) where D is the dimensionality of the feature space
        """

        w, b = v[:-1], v[-1]

        # ITERATIVE VERSION
        # cross_entropy = 0
        # for i in range(N):
        #     z_i = self.LTR[i] * 2.0 - 1.0
        #     x_i = self.DTR[:, i : i + 1]
        #     exponent = -z_i * (numpy.dot(w.T, x_i) + b)
        #     cross_entropy += numpy.logaddexp(0, exponent)

        # VECTORIAL VERSION
        # Z = 2.0 * self.LTR - 1.0
        # S = numpy.dot(w.T, self.DTR) + b  # Scores
        # cross_entropy = numpy.logaddexp(0, -S * Z)
        # regularization_term = (self.λ / 2) * (numpy.linalg.norm(w) ** 2)
        # return regularization_term + cross_entropy.mean()

        # VECTORIAL BALANCED VERSION
        S_non_pulsar = numpy.dot(w.T, self.DTR_non_pulsar) + b
        S_pulsar = numpy.dot(w.T, self.DTR_pulsar) + b
        
        non_pulsar = (1 - self.πT) * numpy.logaddexp(0, -S_non_pulsar * self.Z_non_pulsar).mean()
        pulsar = self.πT * numpy.logaddexp(0, -S_pulsar * self.Z_pulsar).mean()

        regularization_term = (self.λ / 2) * (numpy.linalg.norm(w) ** 2)
        return regularization_term + non_pulsar + pulsar

    def score_samples(self, DTE: numpy.ndarray, quadratic: bool = False) -> numpy.ndarray:
        """
        Assigns a score to each sample in the dataset.

        Args:
            DTE (numpy.ndarray): Test data to predict
            quadratic (bool):    Whether to use a quadratic expanded feature space. Defaults to False.

        Returns:
            numpy.ndarray: Array of scores
        """
        if quadratic:
            DTE = self.features_expansion(DTE)
        return numpy.dot(self.w.T, DTE) + self.b

    @staticmethod
    def predict_samples(scores: numpy.ndarray):
        """
        Predicts the labels for the given scores.

        Args:
            scores (numpy.ndarray): Array of scores for each sample

        Returns:
            numpy.ndarray: Predicted labels
        """
        # Compute scores as posterior log-likelihood ratios
        return numpy.where(scores >= 0, 1, 0)
    
    @staticmethod
    def features_expansion(X: numpy.ndarray) -> numpy.ndarray:
        """
        Compute a quadratic expanded feature space

        Args:
            X (numpy.ndarray): Samples to transform
        Returns:
            numpy.ndarray: Transformed samples
        """
        φ = []
        for i in range(X.shape[1]):
            Xt = vcol(X[:, i])
            vec = numpy.reshape(numpy.dot(Xt, Xt.T), (-1, 1), order='F')
            φ.append(vec)
        return numpy.vstack((numpy.hstack(φ), X))
