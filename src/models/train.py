from data_processing.comparison import minimum_DCF, normalized_DCF, error_rate
from data_processing.analytics import confusion_matrix
from data_processing.utils import extended_data_matrix
from constants import LABELS, APPLICATIONS
from models.lr import LogisticRegression
from models.svm import SVM, KernelType
from models.mvg import MVG, MVGModel
from models.gmm import GMM
from typing import Union
import numpy


# Single Model #


def train_mvg_models(
    DTR: numpy.ndarray,
    LTR: numpy.ndarray,
    DTE: numpy.ndarray,
    LTE: numpy.ndarray,
    labels: list[int],
    application: tuple[float, float, float],
    models: list = MVGModel,
    recalibration: tuple = None,
    **kwargs,
) -> dict[MVGModel, float]:

    class_priors = [application[0], 1 - application[0]]
    trained_models = {model_type: {} for model_type in models}

    if recalibration is not None:
        _, α, β = recalibration
    else:
        α, β = None, None

    for model_type in models:

        trained_models[model_type]["actDCF"] = {}
        trained_models[model_type]["minDCF"] = {}
        trained_models[model_type]["error_rate"] = {}

        # Train model
        model = MVG(model_type, DTR, LTR, labels)

        for π_tilde, _, _ in APPLICATIONS:

            threshold = -numpy.log(π_tilde / (1 - π_tilde))

            # Compute scores for each class
            trained_models[model_type]["scores"] = model.score_samples(DTE, class_priors)[1]

            # Recalibrate scores
            if α is not None and β is not None:
                trained_models[model_type]["scores"] = α * trained_models[model_type]["scores"] + β - numpy.log(0.5 / (1 - 0.5))

            # Predict labels
            predictions = numpy.int32(trained_models[model_type]["scores"] > threshold)
            CM = confusion_matrix(predictions, LTE)

            # Compute error rate
            trained_models[model_type]["error_rate"][π_tilde] = error_rate(predictions, LTE)

            # Compute Actual DCF
            trained_models[model_type]["actDCF"][π_tilde] = normalized_DCF(CM, π_tilde, 1, 1)

            # Compute minimum DCF
            trained_models[model_type]["minDCF"][π_tilde] = minimum_DCF(trained_models[model_type]["scores"], LTE, π_tilde, 1, 1)

    return trained_models


def train_log_reg_model(
    DTR: numpy.ndarray,
    LTR: numpy.ndarray,
    DTE: numpy.ndarray,
    LTE: numpy.ndarray,
    λ: float,
    πT: float,
    quadratic: bool,
    application: tuple[float, float, float],
    recalibration: tuple = None,
    **kwargs,
) -> float:

    trained_model = {}

    if recalibration is not None:
        _, α, β = recalibration
    else:
        α, β = None, None

    # Train model
    model = LogisticRegression(DTR, LTR, λ, πT, quadratic)

    # Compute scores for each class
    scores = model.score_samples(DTE, quadratic)

    # Recalibrate scores
    if α is not None and β is not None:
        scores = α * scores + β - numpy.log(0.5 / (1 - 0.5))

    π_tilde = application[0]

    threshold = -numpy.log(π_tilde / (1 - π_tilde))

    # Predict labels
    predictions = numpy.int32(scores > threshold)
    CM = confusion_matrix(predictions, LTE)

    # Compute error rate
    trained_model["error_rate"] = error_rate(predictions, LTE)

    # Compute Actual DCF
    trained_model["actDCF"] = normalized_DCF(CM, π_tilde, 1, 1)

    # Compute minimum DCF
    trained_model["minDCF"] = minimum_DCF(scores, LTE, π_tilde, 1, 1)

    return trained_model


def train_gmm_models(
    DTR: numpy.ndarray,
    LTR: numpy.ndarray,
    DTE: numpy.ndarray,
    LTE: numpy.ndarray,
    steps: Union[int, dict],
    models: list = MVGModel,
    recalibration: tuple = None,
    **kwargs,
) -> float:

    trained_models = {model_type: {} for model_type in models}

    if recalibration is not None:
        _, α, β = recalibration
    else:
        α, β = None, None

    for model_type in models:

        trained_models[model_type]["actDCF"] = {}
        trained_models[model_type]["minDCF"] = {}
        trained_models[model_type]["error_rate"] = {}

        # Train a model for each class
        model_steps = steps if type(steps) == int else steps[model_type]
        gmm_non_pulsar = GMM(DTR[:, LTR == 0], model_steps, model_type)
        gmm_pulsar = GMM(DTR[:, LTR == 1], model_steps, model_type)

        for π_tilde, _, _ in APPLICATIONS:

            threshold = -numpy.log(π_tilde / (1 - π_tilde))

            # Compute scores for each class
            trained_models[model_type]["scores"] = gmm_pulsar.log_pdf(DTE)[1] - gmm_non_pulsar.log_pdf(DTE)[1]

            # Recalibrate scores
            if α is not None and β is not None:
                trained_models[model_type]["scores"] = α * trained_models[model_type]["scores"] + β - numpy.log(0.5 / (1 - 0.5))

            # Predict labels
            predictions = numpy.int32(trained_models[model_type]["scores"] > threshold)
            CM = confusion_matrix(predictions, LTE)

            # Compute error rate
            trained_models[model_type]["error_rate"][π_tilde] = error_rate(predictions, LTE)

            # Compute Actual DCF
            trained_models[model_type]["actDCF"][π_tilde] = normalized_DCF(CM, π_tilde, 1, 1)

            # Compute minimum DCF
            trained_models[model_type]["minDCF"][π_tilde] = minimum_DCF(trained_models[model_type]["scores"], LTE, π_tilde, 1, 1)

    return trained_models


def train_svm_model(
    DTR: numpy.ndarray,
    LTR: numpy.ndarray,
    DTE: numpy.ndarray,
    LTE: numpy.ndarray,
    K: float,
    C: float,
    kernel_type: KernelType,
    kernel_params: tuple,
    πT: float,
    recalibration: tuple = None,
    application: tuple = None,
    **kwargs,
) -> float:

    if recalibration is not None:
        _, α, β = recalibration
    else:
        α, β = None, None

    kernel, csi = None, K**2
    trained_model = {"actDCF": {}, "minDCF": {}, "error_rate": {}}

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
        scores = svm.polynomial_scores(DTR, DTE, c, d, csi)
    elif kernel_type == KernelType.RBF:
        scores = svm.RBF_scores(DTR, DTE, γ, csi)
    else:
        svm.primal()
        DTE_extended = extended_data_matrix(DTE, K)
        scores = svm.score_samples(DTE_extended)

    # Recalibrate scores
    if α is not None and β is not None:
        scores = α * scores + β - numpy.log(0.5 / (1 - 0.5))

    π_tilde = application[0]

    threshold = -numpy.log(π_tilde / (1 - π_tilde))

    # Predict labels
    predictions = numpy.int32(scores > threshold)
    CM = confusion_matrix(predictions, LTE)

    # Compute error rate
    trained_model["error_rate"] = error_rate(predictions, LTE)

    # Compute Actual DCF
    trained_model["actDCF"] = normalized_DCF(CM, π_tilde, 1, 1)

    # Compute minimum DCF
    trained_model["minDCF"] = minimum_DCF(scores, LTE, π_tilde, 1, 1)

    return trained_model


# K-Fold Cross Validation #


def mvg_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    labels: list[int],
    application: tuple[float, float, float],
    models=MVGModel,
    return_scores: bool = False,
    **kwargs,
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
        models (list[MVGModel]):            MVG Model types to be trained
        return_scores (bool):               Whether to return the scores or minimum DCFs
    """

    minDCF = {}
    class_priors = [application[0], 1 - application[0]]
    fold_scores = {model_type: [] for model_type in models}
    scores = {}

    # K-fold cross validation
    for DTR, LTR, DVAL in zip(training_samples, training_labels, validation_samples):

        for model_type in models:
            # Train model
            model = MVG(model_type, DTR, LTR, labels)
            # Compute scores for each class
            model_scores = model.score_samples(DVAL, class_priors)[1]
            fold_scores[model_type].append(model_scores)

    for model_type in models:

        # Concatenate scores
        scores[model_type] = numpy.hstack(fold_scores[model_type])

        # Compute minimum DCF
        if not return_scores:
            minDCF[model_type] = minimum_DCF(scores[model_type], validation_labels, *application)

    return scores if return_scores else minDCF


def logistic_regression_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    λ: float,
    πT: float,
    quadratic: bool,
    application: tuple[float, float, float],
    return_scores: bool = False,
    **kwargs,
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
        quadratic (bool):                   Whether to use quadratic feature expansion
        application (tuple):                Effective prior, Cost of false positive, Cost of false negative
        return_scores (bool):               Whether to return the scores or minimum DCFs
    """

    fold_scores = []

    # K-fold cross validation
    for DTR, LTR, DVAL in zip(training_samples, training_labels, validation_samples):

        # Train model
        model = LogisticRegression(DTR, LTR, λ, πT, quadratic)

        # Compute scores for each class
        fold_scores.append(model.score_samples(DVAL, quadratic))

    scores = numpy.hstack(fold_scores)

    if not return_scores:
        minDCF = minimum_DCF(scores, validation_labels, *application)

    return scores if return_scores else minDCF


def gmm_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    application: tuple[float, float, float],
    steps: Union[int, dict],
    models: list = MVGModel,
    return_scores: bool = False,
    **kwargs,
):
    """
    K-fold cross validation for GMM models

    Args:
        training_samples (numpy.ndarray):   Training dataset, of shape (K, n, m) where n is the number of features and m is the number of samples in a fold (There are K folds)
        training_labels (numpy.ndarray):    Training labels, of shape (K, m, )
        validation_samples (numpy.ndarray): Validation dataset, of shape (K, n, m)
        validation_labels: (numpy.array):   Validation labels, of shape (m, )
        application (tuple):                Effective prior, Cost of false positive, Cost of false negative
        steps (int | dict):                 Number of steps for the EM algorithm, the number of resulting components is 2**steps
        models (list[MVGModel]):            MVG Model types to be trained
        return_scores (bool):               Whether to return the scores or minimum DCFs
    """

    minDCF = {}
    fold_scores = {model_type: [] for model_type in models}
    scores = {}

    # K-fold cross validation
    for DTR, LTR, DVAL in zip(training_samples, training_labels, validation_samples):

        for model_type in models:
            # Train models
            model_steps = steps if type(steps) == int else steps[model_type]
            gmm_non_pulsars = GMM(DTR[:, LTR == 0], model_steps, model_type)
            gmm_pulsars = GMM(DTR[:, LTR == 1], model_steps, model_type)
            # Compute log likelihood ratios
            model_scores = gmm_pulsars.log_pdf(DVAL)[1] - gmm_non_pulsars.log_pdf(DVAL)[1]
            fold_scores[model_type].append(model_scores)

    for model_type in models:

        scores[model_type] = numpy.hstack(fold_scores[model_type])

        if not return_scores:
            minDCF[model_type] = minimum_DCF(scores[model_type], validation_labels, *application)

    return scores if return_scores else minDCF


def svm_kfold(
    training_samples: numpy.ndarray,
    training_labels: numpy.ndarray,
    validation_samples: numpy.ndarray,
    validation_labels: numpy.ndarray,
    K: float,
    C: float,
    kernel_type: KernelType,
    kernel_params: tuple,
    πT: float,
    application: tuple[float, float, float],
    return_scores: bool = False,
    **kwargs,
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
        return_scores (bool):               Whether to return the scores or minimum DCFs
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

    if not return_scores:
        minDCF = minimum_DCF(scores, validation_labels, *application)

    return scores if return_scores else minDCF


def train_best_models(
    DTR_z_normalized: numpy.ndarray,
    LTR: numpy.ndarray,
    DVAL_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
    DTR_gaussianized: numpy.ndarray,
    DVAL_gaussianized: numpy.ndarray,
    use_kfold: bool,
    recalibration_parameters: dict = None,
):

    # Select training function
    if use_kfold:
        train_MVG, train_LR, train_SVM, train_GMM = mvg_kfold, logistic_regression_kfold, svm_kfold, gmm_kfold
    else:
        train_MVG, train_LR, train_SVM, train_GMM = train_mvg_models, train_log_reg_model, train_svm_model, train_gmm_models

    mvg_name = "MVG Tied Full Covariance"
    lr_name = "Linear Logistic Regression $(\lambda = 10^{-6}, \pi_T = 0.1)$"
    svm_name = "Linear SVM (C = $10^{-1}$, $\pi_T = 0.1$)"
    polynomial_svm_name = "Polynomial Kernel SVM (C = 1, c = 0.1, $\pi_T = 0.5$)"
    rbf_svm_name = "RBF kernel SVM (C=$10^2$, $\gamma=0.1$, $\pi_T = 0.5$))"
    gmm_name = "GMM Full Covariance (8 components)"

    best_models = {
        mvg_name: train_MVG(
            DTR_z_normalized,
            LTR,
            DVAL_z_normalized,
            LVAL,
            LABELS,
            application=(0.5, 1, 1),
            models=[MVGModel.TIED],
            return_scores=True,
            recalibration=recalibration_parameters[mvg_name] if recalibration_parameters else None,
        )[MVGModel.TIED],
        lr_name: train_LR(
            DTR_z_normalized,
            LTR,
            DVAL_z_normalized,
            LVAL,
            1e-6,
            πT=0.1,
            quadratic=False,
            application=(0.5, 1, 1),
            return_scores=True,
            recalibration=recalibration_parameters[lr_name] if recalibration_parameters else None,
        ),
        svm_name: train_SVM(
            DTR_z_normalized,
            LTR,
            DVAL_z_normalized,
            LVAL,
            1.0,
            1e-1,
            KernelType.NO_KERNEL,
            None,
            πT=0.1,
            application=(0.5, 1, 1),
            return_scores=True,
            recalibration=recalibration_parameters[svm_name] if recalibration_parameters else None,
        ),
        polynomial_svm_name: train_SVM(
            DTR_z_normalized,
            LTR,
            DVAL_z_normalized,
            LVAL,
            1.0,
            1,
            KernelType.POLYNOMIAL,
            (2, 0.1),
            πT=0.5,
            application=(0.5, 1, 1),
            return_scores=True,
            recalibration=recalibration_parameters[polynomial_svm_name] if recalibration_parameters else None,
        ),
        rbf_svm_name: train_SVM(
            DTR_gaussianized,
            LTR,
            DVAL_gaussianized,
            LVAL,
            1.0,
            100,
            KernelType.RBF,
            (0.1),
            πT=0.5,
            application=(0.5, 1, 1),
            return_scores=True,
            recalibration=recalibration_parameters[rbf_svm_name] if recalibration_parameters else None,
        ),
        gmm_name: train_GMM(
            DTR_z_normalized,
            LTR,
            DVAL_z_normalized,
            LVAL,
            application=(0.5, 1, 1),
            steps=3,
            models=[MVGModel.MVG],
            return_scores=True,
            recalibration=recalibration_parameters[gmm_name] if recalibration_parameters else None,
        )[MVGModel.MVG],
    }

    return best_models
