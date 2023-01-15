from data_processing.utils import extended_data_matrix
from data_processing.comparison import minimum_DCF
from models.lr import LogisticRegression
from models.svm import SVM, KernelType
from models.mvg import MVG, MVGModel
from models.gmm import GMM
import numpy

# Single Model #


def train_log_reg_model(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    test_samples: numpy.ndarray,
    test_labels: numpy.ndarray,
    λ: float,
    application: tuple[float, float, float],
) -> float:

    # Train model
    model = LogisticRegression(training_samples, training_labels, λ, application[0])

    # Compute scores
    scores = model.score_samples(test_samples)

    # Compute minimum DCF
    minDCF = minimum_DCF(scores, test_labels, *application)

    return minDCF


def train_mvg_models(
    DTR: numpy.ndarray,
    LTR: numpy.ndarray,
    DTE: numpy.ndarray,
    LTE: numpy.ndarray,
    class_priors: list[float],
    labels: list[int],
    application: tuple[float, float, float],
) -> dict[MVGModel, float]:

    minDCFs = {}

    for model_type in MVGModel:

        # Train model
        model = MVG(model_type, DTR, LTR, labels)

        # Compute scores for each class
        model_scores = model.score_samples(DTE, class_priors)

        # # Predict labels
        # predictions = MVG.predict_samples(model_scores)

        # # Compute error rates
        # error_rates[model_type] = error_rate(predictions, LTE)

        # Compute minimum DCF
        minDCFs[model_type] = minimum_DCF(model_scores, LTE, *application)

    return minDCFs


def train_gmm_models(
    DTR: numpy.ndarray,
    LTR: numpy.ndarray,
    DTE: numpy.ndarray,
    LTE: numpy.ndarray,
    class_priors: list[float],
    steps: int,
    application: tuple[float, float, float],
) -> float:

    gmm_minDCF = {}

    for model_type in MVGModel:

        # Train a model for each class
        gmm_non_pulsar = GMM(DTR[:, LTR == 0], steps, model_type)
        gmm_pulsar = GMM(DTR[:, LTR == 1], steps, model_type)

        # Compute scores for each class
        posteriors = [gmm_non_pulsar.log_pdf(DTE)[1], gmm_pulsar.log_pdf(DTE)[1]]
        
        # Compute minimum DCF
        gmm_minDCF[model_type] = minimum_DCF(posteriors, LTE, *application)

    return gmm_minDCF


def train_svm_model(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    test_samples: numpy.ndarray,
    test_labels: numpy.ndarray,
    K: float,
    C: float,
    kernel_type: KernelType,
    kernel_params: tuple,
    application: tuple[float, float, float],
) -> float:

    kernel, csi = None, K**2
    πT = application[0]
    
    if kernel_type == KernelType.POLYNOMIAL:
        d, c = kernel_params
        kernel = SVM.polynomial_kernel(training_samples, training_samples, c, d, csi)
    elif kernel_type == KernelType.RBF:
        γ = kernel_params
        kernel = SVM.RBF_kernel(training_samples, training_samples, γ, csi)

    # Train models
    svm = SVM(training_samples, training_labels, C, K, πT, kernel)
    svm.dual()

    # Compute scores
    if kernel_type == KernelType.POLYNOMIAL:
        scores = svm.polynomial_scores(training_samples, test_samples, c, d, csi)
    elif kernel_type == KernelType.RBF:
        scores = svm.RBF_scores(training_samples, test_samples, γ, csi)
    else:
        svm.primal()
        DTE_extended = extended_data_matrix(test_samples, K)
        scores = svm.score_samples(DTE_extended)

    minDCF = minimum_DCF(scores, test_labels, *application)

    return minDCF


# K-Fold Cross Validation #


def mvg_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    labels: list[int],
    application: tuple[float, float, float],
) -> dict[MVGModel, float]:
    """
    K-fold cross validation for MVG models

    Args:
        training_samples (numpy.ndarray):   Training dataset, of shape (K, n, m) where n is the number of features and m is the number of samples in a fold (There are K folds)
        training_labels (numpy.ndarray):    Training labels, of shape (K, m, )
        validation_samples (numpy.ndarray): Validation dataset, of shape (K, n, m)
        validation_labels: (numpy.array):   Validation labels, of shape (m, )
        labels (list[int]):                 List of labels [0, 1, ..., k-1], where k is the number of classes
        application (tuple):                Effective prior, Cost of false positive, Cost of false negative
    """

    minDCF = {}
    class_priors = [application[0], 1 - application[0]]
    fold_scores = {model_type: [] for model_type in MVGModel}

    # K-fold cross validation
    for DTR, LTR, DVAL in zip(training_samples, training_labels, validation_samples):

        for model_type in MVGModel:
            # Train model
            model = MVG(model_type, DTR, LTR, labels)
            # Compute scores for each class
            model_scores = model.score_samples(DVAL, class_priors)[1]
            fold_scores[model_type].append(model_scores)

    for model_type in MVGModel:

        # Concatenate scores
        scores = numpy.hstack(fold_scores[model_type])

        # Compute error rates
        minDCF[model_type] = minimum_DCF(scores, validation_labels, *application)

    return minDCF


def logistic_regression_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    λ: float,
    πT: float,
    application: tuple[float, float, float],
) -> float:
    """
    K-fold cross validation for Logistic regression model

    Args:
        training_samples (numpy.ndarray):   Training dataset, of shape (K, n, m) where n is the number of features and m is the number of samples in a fold (There are K folds)
        training_labels (numpy.ndarray):    Training labels, of shape (K, m, )
        validation_samples (numpy.ndarray): Validation dataset, of shape (K, n, m)
        validation_labels: (numpy.array):   Validation labels, of shape (m, )
        λ (float):                          Regularization parameter
        πT (float):                         Prior probability of the first class
        application (tuple):                Effective prior, Cost of false positive, Cost of false negative
    """

    fold_scores = []

    # K-fold cross validation
    for DTR, LTR, DVAL in zip(training_samples, training_labels, validation_samples):

        # Train model
        model = LogisticRegression(DTR, LTR, λ, πT)
        # Compute scores for each class
        fold_scores.append(model.score_samples(DVAL))

    scores = numpy.hstack(fold_scores)

    minDCF = minimum_DCF(scores, validation_labels, *application)

    return minDCF


def gmm_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    application: tuple[float, float, float],
    steps: int,
):
    """
    K-fold cross validation for GMM models

    Args:
        training_samples (numpy.ndarray):   Training dataset, of shape (K, n, m) where n is the number of features and m is the number of samples in a fold (There are K folds)
        training_labels (numpy.ndarray):    Training labels, of shape (K, m, )
        validation_samples (numpy.ndarray): Validation dataset, of shape (K, n, m)
        validation_labels: (numpy.array):   Validation labels, of shape (m, )
        application (tuple):                Effective prior, Cost of false positive, Cost of false negative
        steps (int):                        Number of steps for the EM algorithm, the number of resulting components is 2**steps
    """

    minDCF = {}
    fold_scores = {model_type: [] for model_type in MVGModel}

    # K-fold cross validation
    for DTR, LTR, DVAL in zip(training_samples, training_labels, validation_samples):

        for model_type in MVGModel:
            # Train models
            gmm_non_pulsars = GMM(DTR[:, LTR == 0], steps, model_type)
            gmm_pulsars = GMM(DTR[:, LTR == 1], steps, model_type)
            # Compute log likelihood ratios
            model_scores = gmm_pulsars.log_pdf(DVAL)[1] - gmm_non_pulsars.log_pdf(DVAL)[1]
            fold_scores[model_type].append(model_scores)

    for model_type in MVGModel:

        scores = numpy.hstack(fold_scores[model_type])

        minDCF[model_type] = minimum_DCF(scores, validation_labels, *application)

        # # Predict labels
        # predictions = numpy.where(scores >= 0, 1, 0)

        # # Compute error rates
        # error_rates[model_type] = error_rate(predictions, validation_labels)

    return minDCF


def svm_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    K: float,
    C: float,
    kernel_type: KernelType,
    kernel_params: tuple,
    πT : float, 
    application: tuple[float, float, float],
):
    """
    K-fold cross validation for SVM models

    Args:
        training_samples (numpy.ndarray):   Training dataset, of shape (K, n, m) where n is the number of features and m is the number of samples in a fold (There are K folds)
        training_labels (numpy.ndarray):    Training labels, of shape (K, m, )
        validation_samples (numpy.ndarray): Validation dataset, of shape (K, n, m)
        validation_labels: (numpy.array):   Validation labels, of shape (m, )
        K (float):                          Weight of regularization of the SVM bias term
        C (float):                          Regularization parameter
        kernel_type (KernelType):           Type of kernel to use
        kernel_params (tuple):              Parameters of the kernel
        πT (float):                         Prior probability of the first class
        application (tuple):                Effective prior, Cost of false positive, Cost of false negative
    """

    scores = []

    # K-fold cross validation
    for DTR, LTR, DVAL in zip(training_samples, training_labels, validation_samples):

        kernel, csi = None, K**2
        if kernel_type == KernelType.POLYNOMIAL:
            d, c = kernel_params
            kernel = SVM.polynomial_kernel(DTR, DTR, c, d, csi)
        elif kernel_type == KernelType.RBF:
            γ = kernel_params
            kernel = SVM.RBF_kernel(DTR, DTR, γ, csi)

        # Train models
        svm = SVM(DTR, LTR, C, K, πT, kernel)
        svm.dual()

        # Compute scores
        if kernel_type == KernelType.POLYNOMIAL:
            fold_scores = svm.polynomial_scores(DTR, DVAL, c, d, csi)
        elif kernel_type == KernelType.RBF:
            fold_scores = svm.RBF_scores(DTR, DVAL, γ, csi)
        else:
            svm.primal()
            DTE_extended = extended_data_matrix(DVAL, K)
            fold_scores = svm.score_samples(DTE_extended)
        scores.append(fold_scores)

    scores = numpy.hstack(scores)

    minDCF = minimum_DCF(scores, validation_labels, *application)

    return minDCF
