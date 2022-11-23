from data_processing.data_load import load_training, load_features
from data_processing.utils import vcol
import numpy


def compute_analytics(features, samples):
    """ Compute and print: mean, min and max value of each feature """
    print("Feature \t\t\t\t\t│  Mean \t│  Min \t\t│  Max")
    print("─" * 90)
    for i, feature in enumerate(features):
        feature_name = feature + (" " * (40 - len(feature)))
        mean_value = samples[i].mean()
        min_value = samples[i].min()
        max_value = samples[i].max()
        print("{} \t│ {:.2f}  \t│ {:.2f}  \t│ {:.2f}".format(feature_name, mean_value, min_value, max_value))


def pearson_correlation(samples):
    """ Compute the Pearson correlation coefficient between each pair of features """
    return numpy.absolute(numpy.corrcoef(samples))


def empirical_mean(array: numpy.ndarray) -> numpy.ndarray:
    """ Compute the empirical mean of a dataset """
    return vcol(array.mean(1))


def covariance_matrix(D: numpy.ndarray) -> numpy.ndarray:
    """ Compute the covariance matrix for a dataset """
    # Compute mean foreach column (feature)
    mu = empirical_mean(D)
    # Compute the 0-mean matrix (centered data)
    DC = D - mu
    # Compute covariance matrix
    C = numpy.dot(DC, DC.T) / float(D.shape[1])
    return C


def between_class_covariance_matrix(D: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray) -> numpy.ndarray:
    """
    Compute the between-class covariance matrix of a dataset D

    Args:
        D (numpy.ndarray): The entire dataset
        labels (numpy.ndarray): Labels of each sample
        classes (numpy.ndarray): List of all distinct labels
        
    Returns:
        numpy.ndarray: The between-class covariance matrix
    """
    mu = D.mean(axis=1)  # Dataset mean
    N = float(D.shape[1])  # Number of samples
    SB = 0
    for c in classes:
        samples_of_class_c = D[:, labels == c]
        nc = float(samples_of_class_c.shape[1])  # Number of samples in class c
        mu_c = samples_of_class_c.mean(axis=1)  # Mean of class c
        m = vcol(mu_c - mu)
        # Contribution of class c to the between-class covariance matrix
        SB += nc * numpy.dot(m, m.T)
    return SB / N


def within_class_covariance_matrix(D: numpy.ndarray, labels: numpy.ndarray, classes: numpy.ndarray) -> numpy.ndarray:
    """
    Compute the between-class covariance matrix of a dataset D

    Args:
        D (numpy.ndarray): The entire dataset
        labels (numpy.ndarray): Labels of each sample
        classes (numpy.ndarray): List of all distinct labels
        
    Returns:
        numpy.ndarray: The within-class covariance matrix
    """
    N = float(D.shape[1])  # Number of samples
    SW = 0
    for c in classes:
        samples_of_class_c = D[:, labels == c]
        nc = float(samples_of_class_c.shape[1])  # Number of samples in class c
        SW += nc * covariance_matrix(samples_of_class_c)
    return SW / N


def logpdf_GAU_ND(X: numpy.ndarray, mu: numpy.ndarray, C: numpy.ndarray) -> numpy.ndarray:
    """
    Compute the log-density of a multivariate Gaussian distribution for all samples

    Args:
        X (numpy.ndarray):  Original dataset, of shape (n, m) where n is the number of features and m is the number of samples
        mu (numpy.ndarray): Mean of the MVG distribution, it has shape (n, 1)
        C (numpy.ndarray):  Covariance matrix of the MVG distribution, it has shape (n, n)

    Returns:
        numpy.ndarray: the log-density of the MVG distribution computed for each sample, it has shape (m, 1)
    """
    covariance_inverse = numpy.linalg.inv(C)
    M = X.shape[0]
    # the absolute value of the log-determinant is the second value
    determinant_logarithm = numpy.linalg.slogdet(C)[1]
    Y = numpy.array([])
    for i in range(X.shape[1]):
        x = X[:, i : i + 1]
        centered_sample = x - mu
        log_density = (
            -(M / 2 * numpy.log(2 * numpy.pi))
            - (1 / 2 * determinant_logarithm)
            - (1 / 2 * numpy.dot(centered_sample.T, numpy.dot(covariance_inverse, centered_sample)))
        )
        Y = numpy.append(Y, log_density)
    return Y.ravel()


def z_normalization(features) -> numpy.ndarray:
    """ Compute the z-normalization of a set of features """
    z_scores = numpy.array([(feature - feature.mean()) / st_dev(feature) for feature in features])
    return z_scores


def st_dev(samples: numpy.ndarray) -> float:
    """ Compute the standard deviation of a set of samples """
    mu = samples.mean()
    differences = [(value - mu) ** 2 for value in samples]
    sum_of_differences = sum(differences)
    standard_deviation = (sum_of_differences / (len(samples) - 1)) ** 0.5
    return standard_deviation


if __name__ == "__main__":

    # Load data from file
    samples, labels = load_training()
    features = load_features()

    compute_analytics(features, samples)
