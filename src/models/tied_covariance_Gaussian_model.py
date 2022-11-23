import numpy
import scipy
from data_processing.utils import vrow, vcol
from data_processing.analytics import empirical_mean, covariance_matrix, logpdf_GAU_ND

## Tied covariance Gaussian Classifier ##
def train_tied_covariance_gaussian_model(DTR, LTR, DTE, class_priors):
    mu = []
    C = numpy.zeros((DTR.shape[0], DTR.shape[0]))
    total_samples = DTR.shape[1]
    for c in range(2):
        data = DTR[:, LTR == c]
        # The ML solution for the mean parameters is the same as the MVG model
        mu.append(empirical_mean(data))
        # the ML solution for the covariance matrix is given by the empirical within-class covariance matrix
        C += data.shape[1] * covariance_matrix(data)
    C = (1 / total_samples) * C

    log_density = S = [logpdf_GAU_ND(DTE, mu[c], C) for c in range(2)] 
    log_prior_probabilities = vcol(numpy.log(class_priors))
    logSJoint = log_density + log_prior_probabilities # matrix of joint densities
    #  log-sum-exp trick  #
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0)) # log-marginal

    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)

    assigned_labels = numpy.argmax(SPost, axis=0)

    return assigned_labels