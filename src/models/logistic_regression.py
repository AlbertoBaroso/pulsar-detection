import scipy.optimize
import numpy


# LOGISTIC REGRESSION #


class LogisticRegression:
    def __init__(self, DTR: numpy.ndarray, LTR: numpy.ndarray, λ: float):
        """
        Args:
            DTR (numpy.ndarray): Training dataset
            LTR (numpy.ndarray): Labels for the training dataset
            λ (float): Regularization parameter
        """
        N = DTR.shape[0]
        self.DTR = DTR
        self.LTR = LTR
        self.λ = λ

        # Train the model
        x, _f, _d = scipy.optimize.fmin_l_bfgs_b(func=self.logreg_obj, x0=numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
        self.w, self.b = x[:N], x[-1]

    def logreg_obj(self, v: numpy.ndarray):
        """
        Objective function for the logistic regression problem.

        Args:
            v (numpy.ndarray): array with shape (D+1,) where D is the dimensionality of the feature space
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
        Z = 2.0 * self.LTR - 1.0
        S = numpy.dot(w.T, self.DTR) + b  # Scores
        cross_entropy = numpy.logaddexp(0, -S * Z)

        regularization_term = (self.λ / 2) * (numpy.linalg.norm(w) ** 2)
        return regularization_term + cross_entropy.mean()

    def score_samples(self, DTE: numpy.ndarray) -> numpy.ndarray:
        """
        Assigns a score to each sample in the dataset.

        Args:
            DTE (numpy.ndarray): Test data to predict

        Returns:
            numpy.ndarray: Array of scores
        """
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
