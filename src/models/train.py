from models.logistic_regression_model import train_logistic_regression_model
from data_processing.error import error_rate
from models.mvg import MVG
import numpy

# TODO: Each time you split the training set in K folds you have to compute a new ranking (or reduced feature space in PCA), and transform each sample in DTE with that ranking. If you gaussianize or apply PCA to the whole training set before K-fold, you will bias your results, since unseen data will no longer be truly unseen


def train_mvg_models(DTR, LTR, DTE, LTE, class_priors):

    # MVG Models
    models = [
        ("MVG", MVG.train_multivariate_gaussian, False),
        ("Naive Bayes MVG", MVG.train_naive_bayes_mvg, False),
        ("Tied MVG", MVG.train_tied_covariance_mvg, True),
        ("Tied Naive Bayes MVG", MVG.train_tied_covariance_naive_bayes_mvg, True),
    ]
    labels = sorted(set(LTR) | set(LTE)) # Get ordered list of labels [0, 1, ..]
    error_rates = []

    for model_name, train_model, is_tied in models:
        # Train models
        µ, Σ = train_model(DTR, LTR, labels)
        # Compute scores
        scores = MVG.score_samples(DTE, µ, Σ, class_priors, labels, tied=is_tied)
        # Predict labels
        predictions = MVG.predict_samples(scores)
        # Compute error rates
        error_rates.append(error_rate(predictions, LTE))

    return error_rates


def train_log_reg_model(training_samples, training_labels, test_samples, test_labels, λ):

    # Train models & Predict labels
    predictions_logisitc_regression = train_logistic_regression_model(training_samples, training_labels, test_samples, λ)

    predictions_logisitc_regression = numpy.array([predictions_logisitc_regression])

    test_labels = numpy.array([test_labels])

    # Compute error rates
    logisitc_regression_error = error_rate(predictions_logisitc_regression, test_labels)

    return logisitc_regression_error
