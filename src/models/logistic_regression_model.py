import numpy
import scipy.optimize
import sys
sys.path.append("./")
from data_processing.utils import vcol 
from data_processing.error import error_rate 


# LOGISTIC REGRESSION

# DTR training data
# LTR training labels
# DTE test data
# λ is a hyper-parameter
def train_logistic_regression_model(DTR, LTR, DTE, λ):

    # v is and array with shape (D+1,)
    # D is the dimensionality of the feature space
    # λ is a hyper-parameter
    def logreg_obj(v, DTR, LTR, λ):
        M = DTR.shape[0]
        N = DTR.shape[1]
        w, b = vcol(v[0:M]), v[-1]
        cross_entropy = 0
        for i in range(N):
            c_i = LTR[i]
            x_i = DTR[:, i:i+1]
            z_i = 2.0 * c_i - 1.0
            s = numpy.dot(w.T, x_i) + b
            # numpy.logaddexp used to sum e^a + e^b without numerical issues, == (1 + e^(-z_i*...))
            cross_entropy += numpy.logaddexp(0, - s * z_i)
        regularization_term = (λ / 2) * (numpy.linalg.norm(w) ** 2)
        return regularization_term + 1/N * cross_entropy
        

    # numerical solver to minimize the logreg_obj function
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj, iprint=1, x0=numpy.zeros(DTR.shape[0] + 1), args=(DTR, LTR, λ), approx_grad=True)

    # Compute scores as posterior log-likelihood ratios:
    w = x[0:DTR.shape[0]]
    b = x[-1]
    predicted = []
    for i in range(DTE.shape[1]):
        x_t = DTE[:, i:i+1]
        score_x_t = numpy.dot(w.T, x_t) + b
        label_x_t = 1 if score_x_t >= 0 else 0
        predicted.append(label_x_t)
        
    return predicted