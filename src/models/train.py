import numpy
import sys
from os.path import exists
sys.path.append("./")
from data_processing.dimensionality_reduction import lda, pca
from data_processing.error import error_rate
from naive_bayes_Gaussian_model import train_naive_bayes_gaussian_model
from multivariate_Gaussian_model import train_multivariate_gaussian_model
from tied_covariance_Gaussian_model import train_tied_covariance_gaussian_model
from tied_naive_bayes_Gaussian_model import train_tied_naive_bayes_gaussian_model
from logistic_regression_model import train_logistic_regression_model
from data_processing.data_load import load_training, load_test
from data_processing.gaussianization import gaussianize_test_samples,  gaussianize_training_samples


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

    # Data load & preprocessing #
    
    training_samples, training_labels = load_training()
    test_samples, test_labels = load_test()
    
    gaussianized_training_file = "../data/processed/gaussianized_training.npy"
    gaussianized_test_file = "../data/processed/gaussianized_test.npy"
    pca_training_file = "../data/processed/pca_training.npy"
    pca_test_file = "../data/processed/pca_test.npy"
    lda_training_file = "../data/processed/lda_training.npy"
    lda_test_file = "../data/processed/lda_test.npy"
    
    if  exists(gaussianized_training_file) and exists(gaussianized_test_file) and \
        exists(pca_training_file) and exists(pca_test_file) and \
        exists(lda_training_file) and exists(lda_test_file):
        # If pre-processed data exists, load it from file
        gaussianized_training_samples = numpy.load(gaussianized_training_file)
        gaussianized_test_samples = numpy.load(gaussianized_test_file)
        pca_training_samples = numpy.load(pca_training_file)
        pca_test_samples = numpy.load(pca_test_file)
        lda_training_samples = numpy.load(lda_training_file)
        lda_test_samples = numpy.load(lda_test_file)
    else:
        # Otherwise, process the data and save it to file
        gaussianized_training_samples = gaussianize_training_samples(training_samples)
        numpy.save(gaussianized_training_file, gaussianized_training_samples)
        gaussianized_test_samples = gaussianize_test_samples(test_samples, training_samples)
        numpy.save(gaussianized_test_file, gaussianized_test_samples)
        
        pca_training_samples = pca(training_samples, 4, training_labels)
        numpy.save(pca_training_file, pca_training_samples)
        pca_test_samples = pca(test_samples, 4, test_labels)
        numpy.save(pca_test_file, pca_test_samples)

        lda_training_samples = lda(training_samples, training_labels, 1)
        numpy.save(lda_training_file, lda_training_samples)
        lda_test_samples = lda(test_samples, test_labels, 1)
        numpy.save(lda_test_file, lda_test_samples)

    
    # TODO: ZERO MEAN DATA
    # TODO


    ##############
    # MVG MODELS #
    ##############

    class_priors_vec = [[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]
    # for class_priors in class_priors_vec:

        # Raw features #

   #    # mvg_error,  naive_bayes_error,  mvg_tied_cov_error,  tied_naive_bayes_error = train_mvg_models(training_samples, training_labels, test_samples, test_labels, class_priors)

    #     # print("# MVG ERROR RATES, priors: {} #".format(class_priors))
    #     # print("Multivariate Gaussian Model: ", mvg_error)
    #     # print("Naive bayes Model: ", naive_bayes_error)
    #     # print("Tied covariance Model: ", mvg_tied_cov_error)
    #     # print("Tied naive bayes Model: ", tied_naive_bayes_error)

        # LDA #
        
    #     # mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error = train_models(lda_training_samples, training_labels, lda_test_samples, test_labels, class_priors)

    #     # print("# LDA ERROR RATES, priors: {} #".format(class_priors))
    #     # print("Multivariate Gaussian Model: ", mvg_error)
    #     # print("Naive bayes Model: ", naive_bayes_error)
    #     # print("Tied covariance Model: ", mvg_tied_cov_error)
    #     # print("Tied naive bayes Model: ", tied_naive_bayes_error)

        # PCA #
        
    #     # mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error = train_models(pca_training_samples, training_labels, pca_test_samples, test_labels, class_priors)

    #     # print("# PCA ERROR RATES, priors: {} #".format(class_priors))
    #     # print("Multivariate Gaussian Model: ", mvg_error)
    #     # print("Naive bayes Model: ", naive_bayes_error)
    #     # print("Tied covariance Model: ", mvg_tied_cov_error)
    #     # print("Tied naive bayes Model: ", tied_naive_bayes_error)
    
        # GAUSSIANIZED FEATURES #
        
        # mvg_error,  naive_bayes_error,  mvg_tied_cov_error,  tied_naive_bayes_error = train_mvg_models(gaussianized_training_samples, training_labels, gaussianized_test_samples, test_labels, class_priors)
        
        # print("# GAUSSIANIZED FEATURES ERROR RATES, priors: {} #".format(class_priors))
        # print("Multivariate Gaussian Model: ", mvg_error)
        # print("Naive bayes Model: ", naive_bayes_error)
        # print("Tied covariance Model: ", mvg_tied_cov_error)
        # print("Tied naive bayes Model: ", tied_naive_bayes_error)
        
        

    #######################
    # Logistic Regression #
    #######################

    log_reg_λ_vec = [0, 0.0001, 0.01, 0.1]
    log_reg_raw_results = []
    log_reg_pca_results = []
    log_reg_lda_results = []
    log_reg_gaussianized_results = []
    
    # for λ in log_reg_λ_vec:
        
    #     # Raw features #
    #     logisitc_regression_error = train_log_reg_model(training_samples, training_labels, test_samples, test_labels, λ)
    #     log_reg_raw_results.append((λ, logisitc_regression_error))
    
    #     # PCA #
    #     logisitc_regression_error = train_log_reg_model(pca_training_samples, training_labels, pca_test_samples, test_labels, λ)
    #     log_reg_pca_results.append((λ, logisitc_regression_error))
    
    #     # LDA #
    #     logisitc_regression_error = train_log_reg_model(lda_training_samples, training_labels, lda_test_samples, test_labels, λ)
    #     log_reg_lda_results.append((λ, logisitc_regression_error))
    
    #     # GAUSSIANIZED FEATURES #
    #     logisitc_regression_error = train_log_reg_model(gaussianized_training_samples, training_labels, gaussianized_test_samples, test_labels, λ)
    #     log_reg_gaussianized_results.append((λ, logisitc_regression_error))
            
    # print("# LOGISTIC REGRESSION ERROR RATES #")
    # for λ, logisitc_regression_error in log_reg_raw_results:
    #     print("LogReg RAW with λ = {} : {}".format(λ, logisitc_regression_error))
    # for λ, logisitc_regression_error in log_reg_pca_results:
    #     print("LogReg PCA with λ = {} : {}".format(λ, logisitc_regression_error))
    # for λ, logisitc_regression_error in log_reg_lda_results:
    #     print("LogReg LDA with λ = {} : {}".format(λ, logisitc_regression_error))
    # for λ, logisitc_regression_error in log_reg_gaussianized_results:
    #     print("LogReg GAUSSIANIZED with λ = {} : {}".format(λ, logisitc_regression_error))
