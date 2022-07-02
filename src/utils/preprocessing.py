# Reshape a one-dimensional array to a column array
def vcol(array):
    return array.reshape((array.size, 1))

# Reshape a column array to a row array
def vrow(array):
    return array.reshape((1, array.size))

## DIMENSIONALITY  REDUCTION ##
import numpy

# Compute empirical mean
def empirical_mean(array):
    return vcol(array.mean(1))

def covariance_matrix(D):
    # Compute mean foreach column (feature)
    mu = empirical_mean(D)
    # Compute the 0-mean matrix (centered data)
    DC = D - mu
    # Compute covariance matrix
    C = numpy.dot(DC, DC.T) / float(D.shape[1])
    return C