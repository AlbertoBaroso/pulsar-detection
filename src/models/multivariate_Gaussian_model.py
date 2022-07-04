import numpy
import sys
sys.path.append("./")
from utils.preprocessing import vrow, vcol
from utils.analytics import empirical_mean, covariance_matrix
from scipy import special

# MULTIVARIATE GAUSSIAN DISTRIBUTION
# Compute log-density for a sample set of samples X
#  M    : Size of the feature vector X
#  X    : M × N matrix of samples x (It's a column vector)
#  mu   : numpy array of shape (M, 1) (It's a column vector)
#  C    : numpy array of shape (M, M) (It's a square matrix) representing the covariance matrix Σ
# Returns Y, the vector of log-densities
def logpdf_GAU_ND(X, mu, C):
    covariance_inverse = numpy.linalg.inv(C)
    M = X.shape[0]
    # the absolute value of the log-determinant is the second value
    determinant_logarithm = numpy.linalg.slogdet(C)[1]
    Y = numpy.array([])
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        centered_sample = x - mu
        log_density = - (M/2 * numpy.log(2 * numpy.pi)) - (1/2 * determinant_logarithm) - (
            1/2 * numpy.dot(centered_sample.T, numpy.dot(covariance_inverse, centered_sample)))
        Y = numpy.append(Y, log_density)
    return Y.ravel()


# Compute the density for a matrix of samples X
def pdf(X, mu, C):
    # Takes the exponent of the logarithm, so just the density not the log-density
    return numpy.exp(logpdf_GAU_ND(X, mu, C))

## Multivariate Gaussian Classifier ##

# the classifier assumes that samples of each class c ∈ {0, 1} can be modeled as samples of a multivariate
# Gaussian distribution with class-dependent mean and covariance matrices, the solution is given by
# the empirical mean and covariance matrix of each class
def train_multivariate_gaussian_model(DTR, LTR, DTE, class_priors):
    mu = []
    C = []
    for c in range(2):
        data = DTR[:, LTR == c]
        mu.append(empirical_mean(data))
        C.append(covariance_matrix(data))

    # The final goal is to compute class posterior probabilities P(c|x), we do that in 3 steps:

    # 1st step: For each sample x, compute the likelihoods    
    # S: Score matrix, S[i, j] = class conditional probability for sample j given class i
    S = [pdf(DTE, mu[c], C[c]) for c in range(2)] 

    # 2nd step: compute class posterior probabilities combining the score matrix with prior information
    # This requires multiplying each row of S by the prior probability of the corresponding class (1/2).
    SJoint = S * vcol(numpy.array(class_priors)) # Joint likelihood for the samples and the corresponding class

    # 3rd step: Compute class posterior probabilities
    SMarginal = vrow(SJoint.sum(0)) # compute the marginal densities (i.e. for each sample sum the values for each class: SJoint[0, j] + SJoint[1, j] + SJoint[2, j])
    SPost = SJoint / SMarginal # This broadcasts SMarginal so that each element of SJoint is divided by the corresponding element in SMarginal

    # Get the class for each sample, the index of the row with highest SPost value
    assigned_labels = numpy.argmax(SPost, axis=0)
    
    log_density = S = [logpdf_GAU_ND(DTE, mu[c], C[c]) for c in range(2)] 
    log_prior_probabilities = vcol(numpy.log(class_priors))
    logSJoint = log_density + log_prior_probabilities # matrix of joint densities
    #  log-sum-exp trick  #
    logSMarginal = vrow(special.logsumexp(logSJoint, axis=0)) # log-marginal

    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)

    assigned_labels = numpy.argmax(SPost, axis=0)
    return assigned_labels