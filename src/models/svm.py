from enum import Enum
import numpy
import scipy
import scipy.optimize
import scipy.special
from data_processing.utils import vcol, vrow, extended_data_matrix


class KernelType(Enum):
    NO_KERNEL = "Linear SVM"
    POLYNOMIAL = "Polynomial kernel SVM"
    RBF = "RBF kernel SVM"


class SVM:
    def __init__(self, DTR: numpy.ndarray, LTR: numpy.ndarray, C: float, K: float, πT: float = None, kernel: numpy.ndarray = None) -> None:
        """
        Train Support Vector Machine model

        Args:
            DTR (numpy.ndarray):               Data matrix
            LTR (numpy.ndarray):               Label vector
            C (float):                         Regularization parameter
            K (float):                         Regularization parameter
            πT (float, optional):              Prior for the first class. Defaults to None.
            kernel (numpy.ndarray, optional):  Kernel function. Defaults to None.
        """

        N = DTR.shape[1]  # Number of samples
        Z = 2 * LTR - 1  # Convert labels to {-1, 1}
        X̂ = numpy.vstack((DTR, numpy.full((1, N), K)))  # The augmented data matrix

        self.kernel = kernel
        kernel = kernel if kernel is not None else numpy.dot(X̂.T, X̂)
        self.Ĥ = vcol(Z) * vrow(Z) * kernel

        self.πT = πT
        self.C = C
        self.Z = Z
        self.N = N
        self.X̂ = X̂

    def primal(self) -> numpy.ndarray:
        """
        Solve the SVM problem using the primal formulation
        """

        ŵ_star = numpy.dot(self.α_star * self.Z, self.X̂.T)
        self.ŵ_star = ŵ_star

        scores = numpy.dot(ŵ_star.T, self.X̂)
        loss = numpy.maximum(numpy.zeros(scores.shape), 1 - self.Z * scores).sum()
        return 0.5 * numpy.linalg.norm(ŵ_star) ** 2 + self.C * loss

    @staticmethod
    def dual_formulation(α: numpy.ndarray, Ĥ: numpy.ndarray, N: int) -> tuple:
        ones = numpy.ones((N, 1))
        Ĥα = numpy.dot(Ĥ, vcol(α))
        Ĵ_dual_loss = -0.5 * numpy.dot(α.T, Ĥα) + numpy.dot(α.T, ones)
        gradient = -numpy.reshape(Ĥα - ones, (N,))
        return -Ĵ_dual_loss, -gradient

    def dual(self) -> tuple:
        """
        Compute SVM solution using the dual formulation

        Returns:
            tuple[Any, float]:
                - α_star: optimal value of the parameters
                - Jmax: maximum value of the objective function
        """
        if self.πT is None:
            constraints = [(0, self.C)] * self.N  # Constraint min and max values of each αi: 0 ≤ αi ≤ C
        else:
            πTemp = len(self.Z[self.Z == 1]) / self.N  # Empirical priors
            πFemp = len(self.Z[self.Z == -1]) / self.N
            Ct = self.C * self.πT / πTemp  # Costs proportional to the priors
            Cf = self.C * (1 - self.πT) / πFemp
            constraints = [(0, Ct) if self.Z[i] == 1 else (0, Cf) for i in range(self.N)]
        
        α_star, Jmax, _d = scipy.optimize.fmin_l_bfgs_b(
            self.dual_formulation, numpy.zeros(self.N), args=(self.Ĥ, self.N), bounds=constraints
        )
        self.α_star = α_star
        return self.α_star, Jmax

    def score_samples(self, X):
        return numpy.dot(self.ŵ_star.T, X)

    @staticmethod
    def predict_samples(scores):
        """
        Predict the class of a sample x
        """
        return numpy.where(scores >= 0, 1, 0)

    @staticmethod
    def duality_gap(Ĵprimal, Ĵdual):
        """
        Check the duality gap
        """
        return Ĵprimal - Ĵdual

    ### KERNELS ###

    @staticmethod
    def polynomial_kernel(X1, X2, c, d, const):
        kernel = numpy.power(numpy.dot(X1.T, X2) + c, d)
        return kernel + const

    @staticmethod
    def RBF_kernel(X1, X2, γ, const):
        distance = vcol((X1**2).sum(0)) + vrow((X2**2).sum(0)) - 2 * numpy.dot(X1.T, X2)
        kernel = numpy.exp(-γ * distance)
        return kernel + const

    def polynomial_scores(self, DTR, DTE, c, d, const):
        return numpy.dot((vcol(self.α_star) * vcol(self.Z)).T, SVM.polynomial_kernel(DTR, DTE, c, d, const))

    def RBF_scores(self, DTR, DTE, γ, const):
        return numpy.dot((vcol(self.α_star) * vcol(self.Z)).T, SVM.RBF_kernel(DTR, DTE, γ, const))


"""

The SVM decision rule is to assign a pattern to class HT if the score is greater than 0, and to
HF otherwise. However, SVM decisions are not probabilistic, and are not able to account for different 
class priors and mis-classification costs. Bayes decisions thus require either a score postprocessing 
step, i.e. score calibration, or cross-validation to select the optimal threshold for a specific application.

"""
