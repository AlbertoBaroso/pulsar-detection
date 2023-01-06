from data_processing.utils import vcol
import numpy


def compute_analytics(features, samples):
    """Compute and print: mean, min and max value of each feature"""
    print("Feature \t\t\t\t\t│  Mean \t│  Min \t\t│  Max")
    print("─" * 90)
    for i, feature in enumerate(features):
        feature_name = feature + (" " * (40 - len(feature)))
        mean_value = samples[i].mean()
        min_value = samples[i].min()
        max_value = samples[i].max()
        print("{} \t│ {:.2f}  \t│ {:.2f}  \t│ {:.2f}".format(feature_name, mean_value, min_value, max_value))


def pearson_correlation(samples):
    """Compute the Pearson correlation coefficient between each pair of features"""
    return numpy.absolute(numpy.corrcoef(samples))


def empirical_mean(array: numpy.ndarray) -> numpy.ndarray:
    """Compute the empirical mean of a dataset"""
    return vcol(array.mean(1))


def covariance_matrix(D: numpy.ndarray) -> numpy.ndarray:
    """Compute the covariance matrix for a dataset"""
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


def z_normalization(features: numpy.ndarray, comparison_features: numpy.ndarray = None) -> numpy.ndarray:
    """Compute the z-normalization of a set of features"""
    z_scores = []
    for i, feature in enumerate(features):
        mean = feature.mean() if comparison_features is None else comparison_features[i].mean()
        standard_deviation = st_dev(feature) if comparison_features is None else st_dev(comparison_features[i])
        z_scores.append((feature - mean) / standard_deviation)
    return numpy.array(z_scores)


def st_dev(samples: numpy.ndarray) -> float:
    """Compute the standard deviation of a set of samples"""
    mu = samples.mean()
    differences = [(value - mu) ** 2 for value in samples]
    sum_of_differences = sum(differences)
    standard_deviation = (sum_of_differences / (len(samples) - 1)) ** 0.5
    return standard_deviation


def confusion_matrix(predicted_labels: numpy.ndarray, actual_labels: numpy.ndarray, K: int = 2) -> numpy.ndarray:
    """
    Compute the confusion matrix given predicted and actual labels

    Args:
        predicted_labels (numpy.ndarray):   Predicted labels for each sample, it has shape (m, 1)
        actual_labels (numpy.ndarray):      Actual labels of test samples
        K (int):                            Number of classes

    Returns:
        numpy.ndarray:                      Confusion matrix
    """

    CM = numpy.zeros((K, K), dtype=int)
    for predicted in range(K):
        for actual in range(K):
            CM[predicted, actual] = ((predicted_labels == predicted) * (actual_labels == actual)).sum()
    return CM


def fnr_fpr(confusion_matrix: numpy.ndarray) -> tuple[float, float]:
    """
    Compute the false negative rate and false positive rate given the confusion matrix

    Args:
        confusion_matrix (numpy.ndarray): Array of shape (K, K) where K is the number of classes

    Returns:
        tuple[float, float]: False negative rate and false positive rate
    """
    FN = confusion_matrix[0][1]
    TP = confusion_matrix[1][1]
    FP = confusion_matrix[1][0]
    TN = confusion_matrix[0][0]
    FNR = FN / (FN + TP)
    FPR = FP / (FP + TN)
    return FNR, FPR
