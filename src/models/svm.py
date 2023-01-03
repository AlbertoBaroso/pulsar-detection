from enum import Enum
import numpy
import scipy
import scipy.optimize
import scipy.special
from data_processing.utils import vcol, vrow, extended_data_matrix


class KernelType(Enum):
    NO_KERNEL = 0
    POLYNOMIAL = 1
    RBF = 2


class SVM:
    def __init__(self, DTR: numpy.ndarray, LTR: numpy.ndarray, C: float, K: float = 1.0, kernel: numpy.ndarray = None) -> None:
        """
        Train Support Vector Machine model

        Args:
            DTR (numpy.ndarray): Data matrix
            LTR (numpy.ndarray): Label vector
            C (float):
            K (float, optional):
            kernel (numpy.ndarray, optional):
        """

        N = DTR.shape[1]  # Number of samples
        Z = 2 * LTR - 1  # Convert labels to {-1, 1}
        X̂ = numpy.vstack((DTR, numpy.full((1, N), K)))  # The augmented data matrix

        self.kernel = kernel
        kernel = kernel if kernel is not None else numpy.dot(X̂.T, X̂)
        self.Ĥ = vcol(Z) * vrow(Z) * kernel

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
        constraints = [(0, self.C)] * self.N  # Constraint min and max values of each αi: 0 ≤ αi ≤ C
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


if __name__ == "__main__":

    D, L = load_iris_binary()
    (DTR, DTE), (LTR, LTE) = split_db_2to1(D, L)

    params = [
        (1.0, 0.1),
        (1.0, 1.0),
        (1.0, 10.0),
        (10.0, 0.1),
        (10.0, 1.0),
        (10.0, 10),
    ]

    # # Higher values of K correspond to weaker regularization of the SVM bias term

    for K, C in params:
        print(f"### K = {K}, C = {C} ###")

        DTE_extended = extended_data_matrix(DTE, K)
        svm = SVM(DTR, LTR, C, K)
        svm.dual()
        dual = -svm.dual_formulation(svm.α_star, svm.X̂, svm.Z, svm.N)[0]
        primal = svm.primal()
        dual_gap = svm.duality_gap(primal, dual)
        scores = svm.score_samples(DTE_extended)
        predictions = SVM.predict_samples(scores)
        error_rate = 1 - (predictions == LTE).sum() / LTE.size

        print("Primal solution: ", primal)
        print("Dual solution: ", dual)
        print("Dual gap: ", dual_gap)
        print("Error rate: ", error_rate)

    print("___ KERNELS ___")

    params = [
        (0.0, 1.0, KernelType.POLYNOMIAL, (2, 0)),
        (1.0, 1.0, KernelType.POLYNOMIAL, (2, 0)),
        (0.0, 1.0, KernelType.POLYNOMIAL, (2, 1)),
        (1.0, 1.0, KernelType.POLYNOMIAL, (2, 1)),
        (0.0, 1.0, KernelType.RBF, (1.0)),
        (0.0, 1.0, KernelType.RBF, (10.0)),
        (1.0, 1.0, KernelType.RBF, (1.0)),
        (1.0, 1.0, KernelType.RBF, (10.0)),
    ]

    for K, C, kernel_type, kernel_params in params:

        csi = K**2
        if kernel_type == KernelType.POLYNOMIAL:
            d, c = kernel_params
            kernel = SVM.polynomial_kernel(DTR, DTR, c, d, csi)
        else:
            γ = kernel_params
            kernel = SVM.RBF_kernel(DTR, DTR, γ, csi)

        svm = SVM(DTR, LTR, C, K, kernel)

        svm.dual()
        dual = -svm.dual_formulation(svm.α_star, svm.X̂, svm.Z, svm.N, kernel)[0]

        if kernel_type == KernelType.POLYNOMIAL:
            scores = svm.polynomial_scores(DTR, DTE, c, d, csi)
        else:
            scores = svm.RBF_scores(DTR, DTE, γ, csi)

        predictions = SVM.predict_samples(scores)
        error_rate = 1 - ((predictions == LTE).sum() / LTE.size)

        print(f"### K = {K}, C = {C}, kernel = {kernel_type}, kernel_params = {kernel_params} ###")
        print("Dual solution: ", dual)
        print("Error rate: ", error_rate)


"""

The SVM decision rule is to assign a pattern to class HT if the score is greater than 0, and to
HF otherwise. However, SVM decisions are not probabilistic, and are not able to account for different 
class priors and mis-classification costs. Bayes decisions thus require either a score postprocessing 
step, i.e. score calibration, or cross-validation to select the optimal threshold for a specific application.

"""
