import numpy
import sys
sys.path.append("./")
from data_processing.dimensionality_reduction import lda, pca
from data_processing.error import error_rate
from naive_bayes_Gaussian_model import train_naive_bayes_gaussian_model
from multivariate_Gaussian_model import train_multivariate_gaussian_model
from tied_covariance_Gaussian_model import train_tied_covariance_gaussian_model
from tied_naive_bayes_Gaussian_model import train_tied_naive_bayes_gaussian_model
from logistic_regression_model import train_logistic_regression_model
from data_processing.data_load import load_training, load_test


def train_mvg_models(training_samples, training_labels, test_samples, test_labels, class_priors):

    # Train models & Predict labels
    predictions_mvg = train_multivariate_gaussian_model(
        training_samples, training_labels, test_samples, class_priors)
    predictions_naive_bayes = train_naive_bayes_gaussian_model(
        training_samples, training_labels, test_samples, class_priors)
    predictions_tied_covariance = train_tied_covariance_gaussian_model(
        training_samples, training_labels, test_samples, class_priors)
    predictions_tied_naive_bayes = train_tied_naive_bayes_gaussian_model(
        training_samples, training_labels, test_samples, class_priors)
    
    predictions_mvg = numpy.array([predictions_mvg])
    predictions_naive_bayes = numpy.array([predictions_naive_bayes])
    predictions_tied_covariance = numpy.array([predictions_tied_covariance])
    predictions_tied_naive_bayes = numpy.array([predictions_tied_naive_bayes])

    test_labels = numpy.array([test_labels])

    # Compute error rates
    mvg_error = error_rate(predictions_mvg, test_labels)
    naive_bayes_error = error_rate(predictions_naive_bayes, test_labels)
    mvg_tied_cov_error = error_rate(predictions_tied_covariance, test_labels)
    tied_naive_bayes_error = error_rate(
        predictions_tied_naive_bayes, test_labels)
    
    return (mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error)

def train_log_reg_model(training_samples, training_labels, test_samples, test_labels, λ):

    # Train models & Predict labels
    predictions_logisitc_regression = train_logistic_regression_model(
        training_samples, training_labels, test_samples, λ)

    predictions_logisitc_regression = numpy.array(
        [predictions_logisitc_regression])

    test_labels = numpy.array([test_labels])

    # Compute error rates
    logisitc_regression_error = error_rate(
        predictions_logisitc_regression, test_labels)
    
    return logisitc_regression_error


if __name__ == '__main__':

    training_samples, training_labels = load_training()
    test_samples, test_labels = load_test()
    class_priors_vec = [[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]
    log_reg_λ_vec = [0, 0.0001, 0.01, 0.1]

    for class_priors in class_priors_vec:

        ################
        # Raw features #
        ################

        (
            mvg_error, 
            naive_bayes_error, 
            mvg_tied_cov_error, 
            tied_naive_bayes_error,
        ) = train_mvg_models(training_samples, training_labels, test_samples, test_labels, class_priors)

        # print("# MVG ERROR RATES, priors: {} #".format(class_priors))
        # print("Multivariate Gaussian Model: ", mvg_error)
        # print("Naive bayes Model: ", naive_bayes_error)
        # print("Tied covariance Model: ", mvg_tied_cov_error)
        # print("Tied naive bayes Model: ", tied_naive_bayes_error)

        # # LDA
        # lda_training_samples = lda(training_samples, training_labels, 1)
        # lda_test_samples = lda(test_samples, test_labels, 1)
        # mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error = train_models(lda_training_samples, training_labels, lda_test_samples, test_labels, class_priors)

        # print("# LDA ERROR RATES, priors: {} #".format(class_priors))
        # print("Multivariate Gaussian Model: ", mvg_error)
        # print("Naive bayes Model: ", naive_bayes_error)
        # print("Tied covariance Model: ", mvg_tied_cov_error)
        # print("Tied naive bayes Model: ", tied_naive_bayes_error)

        # PCA
        # pca_training_samples = pca(training_samples, 4, training_labels)
        # pca_test_samples = pca(test_samples, 4, test_labels)
        # mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error = train_models(pca_training_samples, training_labels, pca_test_samples, test_labels, class_priors)

        # print("# PCA ERROR RATES, priors: {} #".format(class_priors))
        # print("Multivariate Gaussian Model: ", mvg_error)
        # print("Naive bayes Model: ", naive_bayes_error)
        # print("Tied covariance Model: ", mvg_tied_cov_error)
        # print("Tied naive bayes Model: ", tied_naive_bayes_error)

    log_reg_results = []
    for λ in log_reg_λ_vec:
        logisitc_regression_error = train_log_reg_model(
            training_samples, training_labels, test_samples, test_labels, λ)
        log_reg_results.append((λ, logisitc_regression_error))
    
    print("# LOGISTIC REGRESSION ERROR RATES #")
    for λ, logisitc_regression_error in log_reg_results:
        print("Logistic regression with λ = {} : {}".format(λ, logisitc_regression_error))
