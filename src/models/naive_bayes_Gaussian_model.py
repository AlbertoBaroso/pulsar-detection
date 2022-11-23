import numpy
import scipy
from data_processing.utils import vrow, vcol
from data_processing.analytics import empirical_mean, covariance_matrix, logpdf_GAU_ND


"""
We now consider the Naive Bayes version of the classifier. 
The Naive Bayes version of the MVG is simply a Gaussian classifier 
where the covariance matrices are diagonal.
"""

## Naive Bayes Gaussian Classifier ##
def train_naive_bayes_gaussian_model(DTR, LTR, DTE, class_priors):
    mu = []
    C = []
    for c in range(2):
        data = DTR[:, LTR == c]
        # The ML solution for the mean parameters is the same as the MVG model
        mu.append(empirical_mean(data))
        # the ML solution for the covariance matrices is the diagonal of the ML solution for the MVG model
        C.append(covariance_matrix(data) * numpy.identity(data.shape[0]))
        """ 
            Since the number of features is small, we can adapt the MVG code
            by simply zeroing the out-of-diagonal elements of the MVG ML solution.
            This can be done, for example, multiplying element-wise
            the MVG ML solution with the identity matrix. The rest of the code remains unchanged. 
            If we have large dimensional data, it may be advisable to implement ad-hoc 
            functions to work directly with just
            the diagonal of the covariance matrices (we won't do this in this course)
        """

    log_density = S = [logpdf_GAU_ND(DTE, mu[c], C[c]) for c in range(2)] 
    log_prior_probabilities = vcol(numpy.log(class_priors))
    logSJoint = log_density + log_prior_probabilities # matrix of joint densities
    #  log-sum-exp trick  #
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0)) # log-marginal

    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)

    assigned_labels = numpy.argmax(SPost, axis=0)
    return assigned_labels