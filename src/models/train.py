from models.logistic_regression import LogisticRegression
from data_processing.error import error_rate
from models.mvg import MVG, MVGModel
import numpy

# Single Model #

def train_log_reg_model(training_samples: numpy.ndarray, training_labels: numpy.ndarray, test_samples: numpy.ndarray, test_labels: numpy.ndarray, λ: float):

    # Train models & Predict labels
    model = LogisticRegression(training_samples, training_labels, λ)
    
    # Predict labels
    predictions = model.predict_samples(test_samples)

    # Compute error rates
    logisitc_regression_error = error_rate(predictions, test_labels)

    return logisitc_regression_error

def train_mvg_models(DTR: numpy.ndarray, LTR: numpy.ndarray, DTE: numpy.ndarray, LTE: numpy.ndarray, class_priors: list[float], labels: list[int]):

    error_rates = {}

    for model_type in MVGModel:
        
        # Train model
        model = MVG(model_type, DTR, LTR, labels)
        
        # Compute scores for each class
        model_scores = model.score_samples(DTE, class_priors)
        
        # Predict labels
        predictions = MVG.predict_samples(model_scores)
        
        # Compute error rates
        error_rates[model_type] = error_rate(predictions, LTE)

    return error_rates


# K-Fold Cross Validation #


def mvg_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    class_priors: list[float],
    labels: list[int]
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


def logistic_regression_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    λ: float,
):
    """
    K-fold cross validation for Logistic regression model

    Args:
        training_samples (numpy.ndarray):   Training dataset, of shape (K, n, m) where n is the number of features and m is the number of samples in a fold (There are K folds)
        training_labels (numpy.ndarray):    Training labels, of shape (K, m, )
        validation_samples (numpy.ndarray): Validation dataset, of shape (K, n, m)
        validation_labels: (numpy.array):   Validation labels, of shape (m, )
        λ (float):                          Regularization parameter
    """

    error_rates = {}
    scores = []

    # K-fold cross validation
    for DTR, LTR, DVAL in zip(training_samples, training_labels, validation_samples):

        # Train model
        model = LogisticRegression(DTR, LTR, λ)
        # Compute scores for each class
        scores.append(model.score_samples(DVAL))

    fold_scores = numpy.hstack(scores)
        
    # Predict labels
    predictions = LogisticRegression.predict_samples(fold_scores)
    
    # Compute error rates
    error_rates = error_rate(predictions, validation_labels)

    return error_rates
