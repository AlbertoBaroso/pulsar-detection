from data_processing.analytics import covariance_matrix, empirical_mean
from data_processing.utils import vrow, vcol
from models.mvg import MVG, MVGModel
import scipy.special
import numpy


class GMM:
    def __init__(self, DTR: numpy.ndarray, steps: int, type: MVGModel) -> None:
        """
        Gaussian Mixture Model

        Args:
            DTR (numpy.ndarray):        Training data matrix with shape (D, N), where D is the number of dimensions and N is the number of samples
            steps (int):                Number of steps for the EM algorithm, the number of resulting components is 2**steps
            type (MVGModel):            One of the MVGModel types: MVG, NAIVE, TIED, TIED_NAIVE
        """
        self.X = DTR
        self.diagonal = type == MVGModel.NAIVE or type == MVGModel.TIED_NAIVE
        self.tied = type == MVGModel.TIED or type == MVGModel.TIED_NAIVE
        self.LBG(steps)  # Estimate GMM parameters

    def log_pdf(self, X: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """
        Log-density of a GMM for a set of samples

        Args:
            X (numpy.ndarray): Data matrix with shape (D, N), where D is the number of dimensions and N is the number of samples

        Returns:
            S (numpy.ndarray): Matrix with shape (G, N) containing the log-density of each component for each sample
            log_densities (numpy.ndarray): Vector with shape (N,) containing the log-density of the GMM for each sample
        """
        S = []
        for µ, Σ, w in zip(self.M, self.S, self.w):
            cluster_conditional_density = MVG.logpdf_GAU_ND(X, µ, Σ)
            joint_log_density = cluster_conditional_density + numpy.log(w)
            S.append(joint_log_density)
        log_densities = scipy.special.logsumexp(S, axis=0)
        return S, log_densities

    @staticmethod
    def constrain_eigenvalues(Σ: numpy.ndarray, ψ: float) -> numpy.ndarray:
        """
        Update all eigenvalues of a given covariance matrix so that they are at least ψ

        Args:
            Σ (numpy.ndarray):  Covariance matrix to constrain
            ψ (float):          Minimum eigenvalue

        Returns:
            (numpy.ndarray): Covariance matrix with updated eigenvalues
        """
        U, s, _ = numpy.linalg.svd(Σ)
        s[s < ψ] = ψ
        return numpy.dot(U, vcol(s) * U.T)

    def LBG(self, steps: int, α: float = 0.1, ψ: float = 0.01) -> None:
        """
        LBG algorithm to estimate the parameters of the GMM

        Args:
            steps (int):         Number of LBG steps
            α (float, optional): Displacement factor. Defaults to 0.1
            ψ (float, optional): Minimum eigenvalue. Defaults to 0.01
        """

        # Initialize with a single component

        µ = empirical_mean(self.X)
        Σ = GMM.constrain_eigenvalues(covariance_matrix(self.X), ψ)
        w = 1.0
        self.G = 1
        self.M, self.S, self.w = numpy.array([µ]), numpy.array([Σ]), numpy.array([w])
        self.EM(ψ)

        # Generate new components

        for step in range(1, steps + 1):
            new_M, new_S, new_w = [], [], []
            for µ, Σ, w in zip(self.M, self.S, self.w):
                U, s, Vh = numpy.linalg.svd(Σ)
                d = U[:, 0:1] * s[0] ** 0.5 * α  # Displacement vector
                # Split current component into two new components
                new_M.extend([µ + d, µ - d])
                new_S.extend([Σ, Σ])
                new_w.extend([w / 2, w / 2])
            self.M, self.S, self.w = new_M, new_S, new_w
            self.G = 2 ** step
            self.EM(ψ)

    def EM(self, ψ: float = 0.01) -> None:
        """
        EM algorithm to estimate the parameters of the GMM

        Args:
            X (numpy.ndarray): Data matrix with shape (D, N), where D is the number of dimensions and N is the number of samples
            ψ (float, optional): Minimum eigenvalue. Defaults to 0.01.
        """

        delta_l = 1e-6  # Stop threshold
        new_likelihood = None
        improvement = 0
        N = self.X.shape[1]

        while new_likelihood is None or improvement > delta_l:

            old_likelihood = 0 if new_likelihood is None else new_likelihood

            # E-step:

            S, S_marginal = self.log_pdf(self.X)
            γ_gi_log = S - S_marginal
            γ_gi = numpy.exp(γ_gi_log)  # Posterior probabilities of each sample belonging to each cluster
            new_likelihood = S_marginal.sum() / N

            # M-step

            µ, Σ, w = [], [], []

            for g in range(self.G):
                
                γ_g = γ_gi[g, :]
                Zg = γ_g.sum()
                Fg = (vrow(γ_g) * self.X).sum(1)
                Sg = numpy.dot(self.X, (vrow(γ_g) * self.X).T)

                µ_gt = vcol(Fg / Zg)
                Σ_gt = Sg / Zg - numpy.dot(µ_gt, µ_gt.T)
                w_gt = Zg / N

                if self.diagonal:  # Force diagonal covariance matrices
                    Σ_gt = Σ_gt * numpy.eye(Σ_gt.shape[0])
                if self.tied:
                    Σ_gt = Σ_gt * Zg

                # Constrain eigenvalues of Σ_gt to avoid degenerate solutions
                Σ_gt = GMM.constrain_eigenvalues(Σ_gt, ψ)

                µ.append(µ_gt)
                Σ.append(Σ_gt)
                w.append(w_gt)

            if self.tied:  # Force tied covariance matrices
                Σ = sum(Σ) / N
                Σ = GMM.constrain_eigenvalues(Σ, ψ)
                Σ = [Σ for _ in range(self.G)]

            # Update GMM parameters
            self.M = µ
            self.S = Σ
            self.w = w

            improvement = abs(new_likelihood - old_likelihood)