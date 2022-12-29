from models.logistic_regression import LogisticRegression
from data_processing.error import error_rate
from models.mvg import MVG, MVGModel
from models.gmm import GMM
import numpy

# Single Model #


def train_log_reg_model(
    training_samples: numpy.ndarray, training_labels: numpy.ndarray, test_samples: numpy.ndarray, test_labels: numpy.ndarray, λ: float
):

    # Train models & Predict labels
    model = LogisticRegression(training_samples, training_labels, λ)

    # Predict labels
    predictions = model.predict_samples(test_samples)

    # Compute error rates
    logisitc_regression_error = error_rate(predictions, test_labels)

    return logisitc_regression_error


def train_mvg_models(
    DTR: numpy.ndarray, LTR: numpy.ndarray, DTE: numpy.ndarray, LTE: numpy.ndarray, class_priors: list[float], labels: list[int]
):

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


def train_gmm_models(
    DTR: numpy.ndarray, LTR: numpy.ndarray, DTE: numpy.ndarray, LTE: numpy.ndarray, class_priors: list[float], labels: list[int], steps: int
):

    error_rates = {}

    for model_type in MVGModel:
        
        # Train a model for each class
        gmm_non_pulsar = GMM(DTR[:, LTR == 0], steps, model_type)
        gmm_pulsar = GMM(DTR[:, LTR == 1], steps, model_type)
        # Compute scores for each class
        posteriors = [gmm_non_pulsar.log_pdf(DTE)[1], gmm_pulsar.log_pdf(DTE)[1]]
        # Predict labels
        predicitons = numpy.argmax(posteriors, axis=0)
        # Compute error rates
        error_rate = (LTE != predicitons).sum() / LTE.shape[0]
        print("{:.2f}%".format(error_rate * 100))

    return error_rates


# K-Fold Cross Validation #


def mvg_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    class_priors: list[float],
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


def gmm_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    class_priors: list[float],
    steps: int,
):
    """
    K-fold cross validation for GMM models

    Args:
        training_samples (numpy.ndarray):   Training dataset, of shape (K, n, m) where n is the number of features and m is the number of samples in a fold (There are K folds)
        training_labels (numpy.ndarray):    Training labels, of shape (K, m, )
        validation_samples (numpy.ndarray): Validation dataset, of shape (K, n, m)
        validation_labels: (numpy.array):   Validation labels, of shape (m, )
        class_priors (numpy.ndarray):       List of prior probability of each class
        steps (int):                        Number of steps for the EM algorithm, the number of resulting components is 2**steps
    """

    error_rates = {}
    scores = {model_type: [] for model_type in MVGModel}

    # K-fold cross validation
    for DTR, LTR, DVAL in zip(training_samples, training_labels, validation_samples):

        for model_type in MVGModel:
            # Train models
            gmm_non_pulsars = GMM(DTR[:, LTR == 0], steps, model_type)
            gmm_pulsars = GMM(DTR[:, LTR == 1], steps, model_type)
            # Compute log likelihood ratios
            model_scores = gmm_pulsars.log_pdf(DVAL)[1] - gmm_non_pulsars.log_pdf(DVAL)[1]
            scores[model_type].append(model_scores)

    for model_type in MVGModel:

        fold_scores = numpy.hstack(scores[model_type])

        # Predict labels
        predictions = numpy.where(fold_scores >= 0, 1, 0)

        # Compute error rates
        error_rates[model_type] = error_rate(predictions, validation_labels)

    return error_rates
