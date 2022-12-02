from models.logistic_regression_model import train_logistic_regression_model
from data_processing.validation import kfold
from data_processing.error import error_rate
from models.mvg import MVG, MVGModel
import numpy


def train_mvg_models(DTR, LTR, DTE, LTE, class_priors):

    # MVG Models
    models = [
        ("MVG", MVG.train_multivariate_gaussian2, False),
        ("Naive Bayes MVG", MVG.train_naive_bayes_mvg2, False),
        ("Tied MVG", MVG.train_tied_covariance_mvg2, True),
        ("Tied Naive Bayes MVG", MVG.train_tied_covariance_naive_bayes_mvg2, True),
    ]
    labels = sorted(set(LTR) | set(LTE))  # Get ordered list of labels [0, 1, ..]
    error_rates = []

    for model_name, train_model, is_tied in models:
        # Train models
        µ, Σ = train_model(DTR, LTR, labels)
        # Compute scores
        scores = MVG.score_samples2(DTE, µ, Σ, class_priors, labels, tied=is_tied)
        # Predict labels
        predictions = MVG.predict_samples(scores)
        # Compute error rates
        error_rates.append(error_rate(predictions, LTE))

    return error_rates


def mvg_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.array,
    validation_labels: numpy.array,
    class_priors: list,
    labels: list[int],
):
    """
    K-fold cross validation for MVG models

    Args:
        training_samples (numpy.ndarray):   Training dataset, of shape (K, n, m) where n is the number of features and m is the number of samples in a fold (There are K folds)
        training_labels (numpy.ndarray):    Training labels, of shape (K, m, )
        validation_samples (numpy.ndarray): Validation dataset, of shape (K, n, m)
        validation_labels: (numpy.array):   Validation labels, of shape (m, )
        class_priors (numpy.ndarray):       List of prior probability of each class
        labels (list[int]):                 List of labels [0, 1, ..., k-1], where k is the number of classes
    """

    error_rates = {}
    scores = {model_type: [] for model_type in MVGModel}

    # K-fold cross validation
    for DTR, LTR, DVAL in zip(training_samples, training_labels, validation_samples):

        for model_type in MVGModel:
            # Train model
            model = MVG(model_type, DTR, LTR, labels)
            # Compute scores for each class
            model_scores = model.score_samples(DVAL, class_priors)
            scores[model_type].append(model_scores)

    for model_type in MVGModel:

        fold_scores = numpy.hstack(scores[model_type])
        # Predict labels
        predictions = MVG.predict_samples(fold_scores)
        # Compute error rates
        error_rates[model_type] = error_rate(predictions, validation_labels)

    return error_rates


def train_log_reg_model(training_samples, training_labels, test_samples, test_labels, λ):

    # Train models & Predict labels
    predictions_logisitc_regression = train_logistic_regression_model(training_samples, training_labels, test_samples, λ)

    predictions_logisitc_regression = numpy.array([predictions_logisitc_regression])

    test_labels = numpy.array([test_labels])

    # Compute error rates
    logisitc_regression_error = error_rate(predictions_logisitc_regression, test_labels)

    return logisitc_regression_error
