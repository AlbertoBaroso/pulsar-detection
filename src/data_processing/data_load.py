from data_processing.gaussianization import gaussianize_test_samples, gaussianize_training_samples
from data_processing.dimensionality_reduction import lda, pca
from data_processing.analytics import z_normalization
from data_processing.utils import vcol, project_data
from data_processing.validation import kfold
from os.path import exists
import numpy

from constants import K, PCA_COMPONENTS, LDA_COMPONENTS


def load_features() -> list[str]:
    """Load the names of the features"""
    return [
        "Mean of the integrated profile",
        "Standard deviation of the integrated profile",
        "Excess kurtosis of the integrated profile",
        "Skewness of the integrated profile",
        "Mean of the DM-SNR curve",
        "Standard deviation of the DM-SNR curve",
        "Excess kurtosis of the DM-SNR curve",
        "Skewness of the DM-SNR curve",
    ]


def load_data(dataset: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Read the dataset and return the samples and labels"""

    samples = []
    labels = []

    # Read samples
    with open(dataset, "r") as file:
        try:
            for line in file:

                attributes = line.split(",")

                # Extract first 8 attributes as floats
                sample = vcol(numpy.array([float(attr) for attr in attributes[0:8]]))

                # Retrieve class label
                label = int(attributes[-1].strip())

                samples.append(sample)
                labels.append(label)
        except:
            pass

    return numpy.hstack(samples), numpy.array(labels)


def load_training() -> tuple[numpy.ndarray, numpy.ndarray]:
    """Retrieve training samples"""
    return load_data("../data/Train.txt")


def load_test() -> tuple[numpy.ndarray, numpy.ndarray]:
    """Retrieve test samples"""
    return load_data("../data/Test.txt")


def load_preprocessed(training_samples, training_labels, test_samples, test_labels):

    # Cached data files #
    folder = "../data/processed/"
    kfold_version = "kfold_"
    z_normalized_training_file = "z_normalized_training.npy"
    z_normalized_test_file = "z_normalized_test.npy"
    gaussianized_training_file = "gaussianized_training.npy"
    gaussianized_test_file = "gaussianized_test.npy"
    pca_training_file = "pca_training.npy"
    pca_test_file = "pca_test.npy"
    lda_training_file = "lda_training.npy"
    lda_test_file = "lda_test.npy"
    kfold_training_labels_file = "kfold_training_labels.npy"
    kfold_validation_labels_file = "kfold_validation_labels.npy"
    kfold_raw_training_file = "kfold_raw_training.npy"
    kfold_raw_validation_file = "kfold_raw_validation.npy"

    # If pre-processed data exists, load it from file
    # Otherwise, process the data and save it to file

    # Gaussianized features
    if exists(folder + gaussianized_training_file) and exists(folder + gaussianized_test_file):
        DTR_gaussianized = numpy.load(folder + gaussianized_training_file)
        DTE_gaussianized = numpy.load(folder + gaussianized_test_file)
    else:
        DTR_gaussianized = gaussianize_training_samples(training_samples)
        DTE_gaussianized = gaussianize_test_samples(test_samples, training_samples)
        numpy.save(folder + gaussianized_training_file, DTR_gaussianized)
        numpy.save(folder + gaussianized_test_file, DTE_gaussianized)

    # Z-normalized features
    if exists(folder + z_normalized_training_file) and exists(folder + z_normalized_test_file):
        DTR_z_normalized = numpy.load(folder + z_normalized_training_file)
        DTE_z_normalized = numpy.load(folder + z_normalized_test_file)
    else:
        DTR_z_normalized = z_normalization(training_samples)
        DTE_z_normalized = z_normalization(test_samples, training_samples)
        numpy.save(folder + z_normalized_training_file, DTR_z_normalized)
        numpy.save(folder + z_normalized_test_file, DTE_z_normalized)

    # Principal Component Analysis
    if exists(folder + pca_training_file) and exists(folder + pca_test_file):
        DTR_pca = numpy.load(folder + pca_training_file)
        DTE_pca = numpy.load(folder + pca_test_file)
    else:
        pca_projection_matrix = pca(training_samples, PCA_COMPONENTS)
        DTR_pca = project_data(training_samples, pca_projection_matrix)
        DTE_pca = project_data(test_samples, pca_projection_matrix)
        numpy.save(folder + pca_training_file, DTR_pca)
        numpy.save(folder + pca_test_file, DTE_pca)

    # Linear Discriminant Analysis
    if exists(folder + lda_training_file) and exists(folder + lda_test_file):
        DTR_lda = numpy.load(folder + lda_training_file)
        DTE_lda = numpy.load(folder + lda_test_file)
    else:
        lda_projection_matrix = lda(training_samples, training_labels, LDA_COMPONENTS)
        DTR_lda = project_data(training_samples, lda_projection_matrix)
        DTE_lda = project_data(test_samples, lda_projection_matrix)
        numpy.save(folder + lda_training_file, DTR_lda)
        numpy.save(folder + lda_test_file, DTE_lda)

    # Gaussianized features in K-folds
    DTR_kfold_gaussianized_file = folder + kfold_version + gaussianized_training_file
    DVAL_kfold_gaussianized_file = folder + kfold_version + gaussianized_test_file
    if exists(DTR_kfold_gaussianized_file) and exists(DVAL_kfold_gaussianized_file):
        DTR_kfold_gaussianized = numpy.load(DTR_kfold_gaussianized_file, allow_pickle=True)
        DVAL_kfold_gaussianized = numpy.load(DVAL_kfold_gaussianized_file, allow_pickle=True)
    else:
        DTR_kfold_gaussianized, DVAL_kfold_gaussianized = numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray)
        for i, (DTR, _LTR, DVAL, _LVAL) in enumerate(kfold(training_samples, training_labels, K)):
            DTR_kfold_gaussianized[i] = gaussianize_training_samples(DTR)
            DVAL_kfold_gaussianized[i] = gaussianize_test_samples(DVAL, DTR)
        numpy.save(DTR_kfold_gaussianized_file, DTR_kfold_gaussianized)
        numpy.save(DVAL_kfold_gaussianized_file, DVAL_kfold_gaussianized)

    # Z-normalized features in K-folds
    DTR_kfold_z_normalized_file = folder + kfold_version + z_normalized_training_file
    DVAL_kfold_z_normalized_file = folder + kfold_version + z_normalized_test_file
    if exists(DTR_kfold_z_normalized_file) and exists(DVAL_kfold_z_normalized_file):
        DTR_kfold_z_normalized = numpy.load(DTR_kfold_z_normalized_file, allow_pickle=True)
        DVAL_kfold_z_normalized = numpy.load(DVAL_kfold_z_normalized_file, allow_pickle=True)
    else:
        DTR_kfold_z_normalized, DVAL_kfold_z_normalized = numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray)
        for i, (DTR, _LTR, DVAL, _LVAL) in enumerate(kfold(training_samples, training_labels, K)):
            DTR_kfold_z_normalized[i] = z_normalization(DTR)
            DVAL_kfold_z_normalized[i] = z_normalization(DVAL, DTR)
        numpy.save(DTR_kfold_z_normalized_file, DTR_kfold_z_normalized)
        numpy.save(DVAL_kfold_z_normalized_file, DVAL_kfold_z_normalized)

    # PCA features in K-folds
    DTR_kfold_pca_file = folder + kfold_version + pca_training_file
    DVAL_kfold_pca_file = folder + kfold_version + pca_test_file
    if exists(DTR_kfold_pca_file) and exists(DVAL_kfold_pca_file):
        DTR_kfold_pca = numpy.load(DTR_kfold_pca_file, allow_pickle=True)
        DVAL_kfold_pca = numpy.load(DVAL_kfold_pca_file, allow_pickle=True)
    else:
        DTR_kfold_pca, DVAL_kfold_pca = numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray)
        for i, (DTR, _LTR, DVAL, _LVAL) in enumerate(kfold(training_samples, training_labels, K)):
            pca_projection_matrix = pca(DTR, PCA_COMPONENTS)
            DTR_kfold_pca[i] = project_data(DTR, pca_projection_matrix)
            DVAL_kfold_pca[i] = project_data(DVAL, pca_projection_matrix)
        numpy.save(DTR_kfold_pca_file, DTR_kfold_pca)
        numpy.save(DVAL_kfold_pca_file, DVAL_kfold_pca)

    # LDA features in K-folds
    DTR_kfold_lda_file = folder + kfold_version + lda_training_file
    DVAL_kfold_lda_file = folder + kfold_version + lda_test_file
    if exists(DTR_kfold_lda_file) and exists(DVAL_kfold_lda_file):
        DTR_kfold_lda = numpy.load(DTR_kfold_lda_file, allow_pickle=True)
        DVAL_kfold_lda = numpy.load(DVAL_kfold_lda_file, allow_pickle=True)
    else:
        DTR_kfold_lda, DVAL_kfold_lda = numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray)
        for i, (DTR, LTR, DVAL, LVAL) in enumerate(kfold(training_samples, training_labels, K)): 
            lda_projection_matrix = lda(DTR, LTR, LDA_COMPONENTS)
            DTR_kfold_lda[i] = project_data(DTR, lda_projection_matrix)
            DVAL_kfold_lda[i] = project_data(DVAL, lda_projection_matrix)
        numpy.save(DTR_kfold_lda_file, DTR_kfold_lda)
        numpy.save(DVAL_kfold_lda_file, DVAL_kfold_lda)

    # K-fold raw + labels
    DTR_kfold_raw_file = folder + kfold_raw_training_file
    DVAL_kfold_raw_file = folder + kfold_raw_validation_file
    LTR_kfold_file = folder + kfold_training_labels_file
    LVAL_kfold_file = folder + kfold_validation_labels_file
    if exists(DTR_kfold_raw_file) and exists(DVAL_kfold_raw_file) and exists(LTR_kfold_file) and exists(LVAL_kfold_file):
        DTR_kfold_raw = numpy.load(DTR_kfold_raw_file, allow_pickle=True)
        DVAL_kfold_raw = numpy.load(DVAL_kfold_raw_file, allow_pickle=True)
        LTR_kfold = numpy.load(LTR_kfold_file, allow_pickle=True)
        LVAL_kfold = numpy.load(LVAL_kfold_file, allow_pickle=True)
    else:
        DTR_kfold_raw, DVAL_kfold_raw = numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray)
        LTR_kfold, LVAL_kfold = numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray)
        for i, (DTR, LTR, DVAL, LVAL) in enumerate(kfold(training_samples, training_labels, K)):
            DTR_kfold_raw[i], LTR_kfold[i] = DTR, LTR
            DVAL_kfold_raw[i], LVAL_kfold[i] = DVAL, LVAL
        numpy.save(DTR_kfold_raw_file, DTR_kfold_raw)
        numpy.save(DVAL_kfold_raw_file, DVAL_kfold_raw)
        numpy.save(LTR_kfold_file, LTR_kfold)
        numpy.save(LVAL_kfold_file, LVAL_kfold)

    return (
        DTR_gaussianized,
        DTE_gaussianized,
        DTR_pca,
        DTE_pca,
        DTR_lda,
        DTE_lda,
        DTR_kfold_raw,
        DVAL_kfold_raw,
        DTR_z_normalized,
        DTE_z_normalized,
        DTR_kfold_gaussianized,
        DVAL_kfold_gaussianized,
        DTR_kfold_z_normalized,
        DVAL_kfold_z_normalized,
        DTR_kfold_pca,
        DVAL_kfold_pca,
        DTR_kfold_lda,
        DVAL_kfold_lda,
        LTR_kfold,
        LVAL_kfold,
    )
