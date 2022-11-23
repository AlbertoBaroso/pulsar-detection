from data_processing.gaussianization import gaussianize_test_samples, gaussianize_training_samples
from data_processing.data_load import load_training, load_test
from data_processing.dimensionality_reduction import lda, pca
from data_processing.analytics import z_normalization
from data_processing.utils import split_80_20
from os.path import exists
import numpy

from models.train import train_mvg_models, train_log_reg_model


def preprocessing(training_samples, training_labels, test_samples, test_labels):

    # Cached data files #
    preprocessing_folder = "../data/processed/"
    z_normalized_training_file = preprocessing_folder + "z_normalized_training.npy"
    z_normalized_test_file = preprocessing_folder + "z_normalized_test.npy"
    gaussianized_training_file = preprocessing_folder + "gaussianized_training.npy"
    gaussianized_test_file = preprocessing_folder + "gaussianized_test.npy"
    pca_training_file = preprocessing_folder + "pca_training.npy"
    pca_test_file = preprocessing_folder + "pca_test.npy"
    lda_training_file = preprocessing_folder + "lda_training.npy"
    lda_test_file = preprocessing_folder + "lda_test.npy"

    # If pre-processed data exists, load it from file
    # Otherwise, process the data and save it to file

    # Gaussianized features
    if exists(gaussianized_training_file) and exists(gaussianized_test_file):
        gaussianized_training_samples = numpy.load(gaussianized_training_file)
        gaussianized_test_samples = numpy.load(gaussianized_test_file)
    else:
        gaussianized_training_samples = gaussianize_training_samples(training_samples)
        numpy.save(gaussianized_training_file, gaussianized_training_samples)
        gaussianized_test_samples = gaussianize_test_samples(test_samples, training_samples)
        numpy.save(gaussianized_test_file, gaussianized_test_samples)

    # Z-normalized features
    if exists(z_normalized_training_file) and exists(z_normalized_test_file):
        z_normalized_training_samples = numpy.load(z_normalized_training_file)
        z_normalized_test_samples = numpy.load(z_normalized_test_file)
    else:
        z_normalized_training_samples = z_normalization(training_samples)
        numpy.save(z_normalized_training_file, z_normalized_training_samples)
        z_normalized_test_samples = z_normalization(test_samples)
        numpy.save(z_normalized_test_file, z_normalized_test_samples)

    # Principal Component Analysis
    if exists(pca_training_file) and exists(pca_test_file):
        pca_training_samples = numpy.load(pca_training_file)
        pca_test_samples = numpy.load(pca_test_file)
    else:
        pca_training_samples = pca(training_samples, 4)
        numpy.save(pca_training_file, pca_training_samples)
        pca_test_samples = pca(test_samples, 4)
        numpy.save(pca_test_file, pca_test_samples)

    # Linear Discriminant Analysis
    if exists(lda_training_file) and exists(lda_test_file):
        lda_training_samples = numpy.load(lda_training_file)
        lda_test_samples = numpy.load(lda_test_file)
    else:
        lda_training_samples = lda(training_samples, training_labels, 1)
        numpy.save(lda_training_file, lda_training_samples)
        lda_test_samples = lda(test_samples, test_labels, 1)
        numpy.save(lda_test_file, lda_test_samples)

    return (
        gaussianized_training_samples,
        gaussianized_test_samples,
        pca_training_samples,
        pca_test_samples,
        lda_training_samples,
        lda_test_samples,
        z_normalized_training_samples,
        z_normalized_test_samples,
    )


if __name__ == "__main__":

    # Data load & preprocessing #

    training_samples, training_labels = load_training()
    test_samples, test_labels = load_test()

    # Pre-process data #
    (
        gaussianized_training_samples,
        gaussianized_test_samples,
        pca_training_samples,
        pca_test_samples,
        lda_training_samples,
        lda_test_samples,
        z_normalized_training_samples,
        z_normalized_test_samples,
    ) = preprocessing(training_samples, training_labels, test_samples, test_labels)

    # Split 80-20 #
    training_split_samples, training_split_labels, validation_split_samples, validation_split_labels = split_80_20(
        training_samples, training_labels
    )
    pca_training_split_samples, pca_training_split_labels, pca_validation_split_samples, pca_validation_split_labels = split_80_20(
        pca_training_samples, training_labels
    )
    lda_training_split_samples, lda_training_split_labels, lda_validation_split_samples, lda_validation_split_labels = split_80_20(
        lda_training_samples, training_labels
    )
    (
        gaussianized_training_split_samples,
        gaussianized_training_split_labels,
        gaussianized_validation_split_samples,
        gaussianized_validation_split_labels,
    ) = split_80_20(gaussianized_training_samples, training_labels)

    #######################
    # Logistic Regression #
    #######################

    log_reg_λ_vec = [0, 0.0001, 0.01, 0.1]
    log_reg_raw_results = []
    log_reg_pca_results = []
    log_reg_lda_results = []
    log_reg_gaussianized_results = []

    for λ in log_reg_λ_vec:

        # Raw features #
        logisitc_regression_error = train_log_reg_model(training_samples, training_labels, test_samples, test_labels, λ)
        log_reg_raw_results.append((λ, logisitc_regression_error))

        # PCA #
        logisitc_regression_error = train_log_reg_model(pca_training_samples, training_labels, pca_test_samples, test_labels, λ)
        log_reg_pca_results.append((λ, logisitc_regression_error))

        # LDA #
        logisitc_regression_error = train_log_reg_model(lda_training_samples, training_labels, lda_test_samples, test_labels, λ)
        log_reg_lda_results.append((λ, logisitc_regression_error))

        # GAUSSIANIZED FEATURES #
        logisitc_regression_error = train_log_reg_model(
            gaussianized_training_samples, training_labels, gaussianized_test_samples, test_labels, λ
        )
        log_reg_gaussianized_results.append((λ, logisitc_regression_error))

    print("\n" * 5)
    print("# LOGISTIC REGRESSION ERROR RATES #")
    for λ, logisitc_regression_error in log_reg_raw_results:
        print("LogReg RAW with λ = {} : {}".format(λ, logisitc_regression_error))
    for λ, logisitc_regression_error in log_reg_pca_results:
        print("LogReg PCA with λ = {} : {}".format(λ, logisitc_regression_error))
    for λ, logisitc_regression_error in log_reg_lda_results:
        print("LogReg LDA with λ = {} : {}".format(λ, logisitc_regression_error))
    for λ, logisitc_regression_error in log_reg_gaussianized_results:
        print("LogReg GAUSSIANIZED with λ = {} : {}".format(λ, logisitc_regression_error))

    ##############
    # MVG MODELS #
    ##############

    class_priors_vec = [[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]
    for class_priors in class_priors_vec:

        # Raw features #

        mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error = train_mvg_models(
            training_split_samples, training_split_labels, validation_split_samples, validation_split_labels, class_priors
        )

        print("# MVG ERROR RATES, priors: {} #".format(class_priors))
        print("Multivariate Gaussian Model: ", mvg_error)
        print("Naive bayes Model: ", naive_bayes_error)
        print("Tied covariance Model: ", mvg_tied_cov_error)
        print("Tied naive bayes Model: ", tied_naive_bayes_error)

        # LDA #

        mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error = train_mvg_models(
            lda_training_split_samples, training_split_labels, lda_validation_split_samples, validation_split_labels, class_priors
        )

        print("# LDA ERROR RATES, priors: {} #".format(class_priors))
        print("Multivariate Gaussian Model: ", mvg_error)
        print("Naive bayes Model: ", naive_bayes_error)
        print("Tied covariance Model: ", mvg_tied_cov_error)
        print("Tied naive bayes Model: ", tied_naive_bayes_error)

        # PCA #

        mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error = train_mvg_models(
            pca_training_split_samples, training_split_labels, pca_validation_split_samples, validation_split_labels, class_priors
        )

        print("# PCA ERROR RATES, priors: {} #".format(class_priors))
        print("Multivariate Gaussian Model: ", mvg_error)
        print("Naive bayes Model: ", naive_bayes_error)
        print("Tied covariance Model: ", mvg_tied_cov_error)
        print("Tied naive bayes Model: ", tied_naive_bayes_error)

        # GAUSSIANIZED FEATURES #

        mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error = train_mvg_models(
            gaussianized_training_split_samples, training_split_labels, gaussianized_validation_split_samples, validation_split_labels, class_priors
        )

        print("# GAUSSIANIZED FEATURES ERROR RATES, priors: {} #".format(class_priors))
        print("Multivariate Gaussian Model: ", mvg_error)
        print("Naive bayes Model: ", naive_bayes_error)
        print("Tied covariance Model: ", mvg_tied_cov_error)
        print("Tied naive bayes Model: ", tied_naive_bayes_error)
