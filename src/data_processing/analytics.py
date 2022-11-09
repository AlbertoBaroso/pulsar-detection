from data_processing.data_load import load_training, load_features
from data_processing.utils import vcol
import numpy

# Compute mean, min and max value of each feature
def compute_analytics(features, samples):
    print("Feature \t\t\t\t\t│  Mean \t│  Min \t\t│  Max")
    print("─" * 90)
    for i, feature in enumerate(features):
        feature_name = feature + (" " * (40 - len(feature)))
        mean_value = samples[i].mean()
        min_value = samples[i].min()
        max_value = samples[i].max()
        print("{} \t│ {:.2f}  \t│ {:.2f}  \t│ {:.2f}".format(
            feature_name, mean_value, min_value, max_value))


def pearson_correlation(samples):
    return numpy.absolute(numpy.corrcoef(samples))

# Compute empirical mean
def empirical_mean(array):
    return vcol(array.mean(1))


## DIMENSIONALITY  REDUCTION ##
def covariance_matrix(D):
    # Compute mean foreach column (feature)
    mu = empirical_mean(D)
    # Compute the 0-mean matrix (centered data)
    DC = D - mu
    # Compute covariance matrix
    C = numpy.dot(DC, DC.T) / float(D.shape[1])
    return C


# compute the log-densities for samples X
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


def z_normalization(features):
    z_scores = numpy.array(
        [(feature - feature.mean()) / st_dev(feature) for feature in features])
    return z_scores


def st_dev(samples):
    mu = samples.mean()
    differences = [(value - mu)**2 for value in samples]
    sum_of_differences = sum(differences)
    standard_deviation = (sum_of_differences / (len(samples) - 1)) ** 0.5
    return standard_deviation


if __name__ == '__main__':

    # Load data from file
    samples, labels = load_training()
    features = load_features()

    compute_analytics(features, samples)
