import numpy
import sys
sys.path.append("./")
from utils.data_load import load_training, load_test
from tied_naive_bayes_Gaussian_model import train_tied_naive_bayes_gaussian_model
from tied_covariance_Gaussian_model import train_tied_covariance_gaussian_model
from multivariate_Gaussian_model import train_multivariate_gaussian_model
from naive_bayes_Gaussian_model import train_naive_bayes_gaussian_model
from utils.error import error_rate
from utils.preprocessing import lda, pca

def train_models(training_samples, training_labels, test_samples, test_labels, class_priors):
    
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
    tied_naive_bayes_error = error_rate(predictions_tied_naive_bayes, test_labels)
    return mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error


if __name__ == '__main__':

    training_samples, training_labels = load_training()
    test_samples, test_labels = load_test()
    class_priors_vec = [[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]]
    
    for class_priors in class_priors_vec:
    
        # Raw features
        # mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error = train_models(training_samples, training_labels, test_samples, test_labels, class_priors)

        # print("# ERROR RATES, priors: {} #".format(class_priors))
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
        pca_training_samples = pca(training_samples, 4, training_labels)
        pca_test_samples = pca(test_samples, 4, test_labels)
        mvg_error, naive_bayes_error, mvg_tied_cov_error, tied_naive_bayes_error = train_models(pca_training_samples, training_labels, pca_test_samples, test_labels, class_priors)

        print("# LDA ERROR RATES, priors: {} #".format(class_priors))
        print("Multivariate Gaussian Model: ", mvg_error)
        print("Naive bayes Model: ", naive_bayes_error)
        print("Tied covariance Model: ", mvg_tied_cov_error)
        print("Tied naive bayes Model: ", tied_naive_bayes_error)
