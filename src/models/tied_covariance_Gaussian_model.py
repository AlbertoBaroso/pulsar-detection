import numpy
import scipy
import sys
sys.path.append("./")
from data_processing.utils import vrow, vcol
from data_processing.analytics import empirical_mean, covariance_matrix, logpdf_GAU_ND, pdf

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

    # The final goal is to compute class posterior probabilities P(c|x), we do that in 3 steps:

    # 1st step: For each sample x, compute the likelihoods    
    # S: Score matrix, S[i, j] = class conditional probability for sample j given class i
    S = [pdf(DTE, mu[c], C) for c in range(2)] 

    # 2nd step: compute class posterior probabilities combining the score matrix with prior information 
    # This requires multiplying each row of S by the prior probability of the corresponding class (1/2).
    SJoint = S * vcol(numpy.array(class_priors)) # Joint likelihood for the samples and the corresponding class

    # 3rd step: Compute class posterior probabilities
    SMarginal = vrow(SJoint.sum(0)) # compute the marginal densities (i.e. for each sample sum the values for each class: SJoint[0, j] + SJoint[1, j] + SJoint[2, j])
    SPost = SJoint / SMarginal # This broadcasts SMarginal so that each element of SJoint is divided by the corresponding element in SMarginal

    # Get the class for each sample, the index of the row with highest SPost value
    assigned_labels = numpy.argmax(SPost, axis=0)

    log_density = S = [logpdf_GAU_ND(DTE, mu[c], C) for c in range(2)] 
    log_prior_probabilities = vcol(numpy.log(class_priors))
    logSJoint = log_density + log_prior_probabilities # matrix of joint densities
    #  log-sum-exp trick  #
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0)) # log-marginal

    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)

    assigned_labels = numpy.argmax(SPost, axis=0)

    return assigned_labels