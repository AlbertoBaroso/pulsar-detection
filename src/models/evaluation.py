from constants import APPLICATIONS, K, LABELS, LOG_REG_λ, SVM_K, SVM_C, SVM_c, SVM_γ, GMM_STEPS
from models.train import mvg_kfold, logistic_regression_kfold, svm_kfold, gmm_kfold
from visualization.evaluation_plots import plot_minDCF, bayes_error_plot
from data_processing.dimensionality_reduction import pca
from visualization.print import print_separator
from data_processing.utils import project_data
from data_processing.validation import kfold
from models.svm import KernelType
from models.mvg import MVGModel
import numpy


#########################
# Hyperparameter tuning #
#########################


def PCA_m_selection(training_samples: numpy.ndarray, training_labels: numpy.ndarray) -> None:
    """
    Evaluate different values of the hyperparameter m for PCA

    Args:
        training_samples (numpy.ndarray): Training samples
        training_labels (numpy.ndarray): Training labels for the samples
    """

    print_separator("PCA hyperparameter selection")

    # Test m = 8, 7, 6, 5, 4, 3
    from_components = 8
    to_components = 3

    for components in range(from_components, to_components - 1, -1):

        # Divide training set into K folds
        DTR_kfold_pca, DVAL_kfold_pca = numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray)
        LTR_kfold = numpy.empty(K, dtype=numpy.ndarray)
        for i, (DTR, LTR, DVAL, _LVAL) in enumerate(kfold(training_samples, training_labels, K)):
            # Apply PCA projection
            pca_projection_matrix = pca(DTR, components)
            DTR_kfold_pca[i] = project_data(DTR, pca_projection_matrix)
            DVAL_kfold_pca[i] = project_data(DVAL, pca_projection_matrix)
            LTR_kfold[i] = LTR

        for application in APPLICATIONS:

            # Evaluate Gaussian models
            validated_mvg_models = mvg_kfold(DTR_kfold_pca, LTR_kfold, DVAL_kfold_pca, training_labels, LABELS, application)
            print("# PCA m = {} - {} #".format(components, application))
            for model_type, model_minDCF in validated_mvg_models.items():
                print("{}: {:.3f}".format(model_type, model_minDCF))

    print_separator()


def LR_λ_selection(
    DTR_kfold_gaussianized: numpy.ndarray,
    DTR_kfold_z_normalized: numpy.ndarray,
    LTR_kfold: numpy.ndarray,
    DVAL_kfold_gaussianized: numpy.ndarray,
    DVAL_kfold_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
) -> None:
    """
    Evaluate different values of the hyperparameter λ for Logistic Regression models by plotting the minDCF
    """

    print_separator("LR hyperparameter selection")

    minDCFs = {"z-normalized": {}, "gaussianized": {}}

    for application in APPLICATIONS:

        minDCFs["z-normalized"][application] = []
        minDCFs["gaussianized"][application] = []

        for λ in LOG_REG_λ:

            minDCFs["z-normalized"][application].append(
                logistic_regression_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, λ, 0.5, False, application)
            )
            minDCFs["gaussianized"][application].append(
                logistic_regression_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, λ, 0.5, False, application)
            )

    # Plot Z-NORMALIZED minDCFs
    plot_minDCF("Z-Normalized features", "λ", LOG_REG_λ, minDCFs["z-normalized"], True)

    # Plot GAUSSIANIZED minDCFs
    plot_minDCF("Gaussianized features", "λ", LOG_REG_λ, minDCFs["gaussianized"], True)

    print_separator()


def SVM_C_selection(
    DTR_kfold_gaussianized: numpy.ndarray,
    DTR_kfold_z_normalized: numpy.ndarray,
    LTR_kfold: numpy.ndarray,
    DVAL_kfold_gaussianized: numpy.ndarray,
    DVAL_kfold_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
) -> None:
    """
    Evaluate different values of the hyperparameter C for Support Vector Machines by plotting the minDCF
    """

    πTs = [None, 0.5]
    kernel_params = None
    kernel_type = KernelType.NO_KERNEL

    print_separator(kernel_type.value + ", hyperparameter selection")

    minDCFs = {"z-normalized": {}, "gaussianized": {}}

    for πT in πTs:

        minDCFs["z-normalized"][πT] = {}
        minDCFs["gaussianized"][πT] = {}

        for application in APPLICATIONS:

            minDCFs["z-normalized"][πT][application] = []
            minDCFs["gaussianized"][πT][application] = []

            for C in SVM_C:

                minDCFs["z-normalized"][πT][application].append(
                    svm_kfold(
                        DTR_kfold_z_normalized,
                        LTR_kfold,
                        DVAL_kfold_z_normalized,
                        LVAL,
                        SVM_K,
                        C,
                        kernel_type,
                        kernel_params,
                        πT,
                        application,
                    )
                )
                minDCFs["gaussianized"][πT][application].append(
                    svm_kfold(
                        DTR_kfold_gaussianized,
                        LTR_kfold,
                        DVAL_kfold_gaussianized,
                        LVAL,
                        SVM_K,
                        C,
                        kernel_type,
                        kernel_params,
                        πT,
                        application,
                    )
                )

    # Plot Z-NORMALIZED minDCFs
    plot_minDCF(kernel_type.value + ", Z-Normalized features, Unbalanced", "C", SVM_C, minDCFs["z-normalized"][πTs[0]], True)
    plot_minDCF(kernel_type.value + ", Z-Normalized features, Balanced", "C", SVM_C, minDCFs["z-normalized"][πTs[1]], True)

    # Plot GAUSSIANIZED minDCFs
    plot_minDCF(kernel_type.value + ", Gaussianized features, Unbalanced", "C", SVM_C, minDCFs["gaussianized"][πTs[0]], True)
    plot_minDCF(kernel_type.value + ", Gaussianized features, Balanced", "C", SVM_C, minDCFs["gaussianized"][πTs[1]], True)

    print_separator()


def SVM_C_kernel_selection(
    DTR_kfold_gaussianized: numpy.ndarray,
    DTR_kfold_z_normalized: numpy.ndarray,
    LTR_kfold: numpy.ndarray,
    DVAL_kfold_gaussianized: numpy.ndarray,
    DVAL_kfold_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
    kernel_type: KernelType,
) -> None:
    """
    Evaluate different values of the hyperparameter C and parameters of the kernel for Support Vector Machines by plotting the minDCF
    """

    import datetime

    print_separator(kernel_type.value + ", hyperparameter selection")

    minDCFs = {"z-normalized": {}, "gaussianized": {}}
    application = APPLICATIONS[0]

    if kernel_type == KernelType.RBF:
        kernel_params = [(γ) for γ in SVM_γ]
        param_name = r"$\tilde{\pi} = 0.5) - \gamma$"
    elif kernel_type == KernelType.POLYNOMIAL:
        d = 2  # Quadratic polynomial
        kernel_params = [(d, c) for c in SVM_c]
        param_name = r"$\tilde{\pi}$ = 0.5) - c"

    for params in kernel_params:

        metric = params[-1]
        minDCFs["z-normalized"][metric] = []
        minDCFs["gaussianized"][metric] = []

        for C in SVM_C:

            minDCFs["z-normalized"][metric].append(
                svm_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, SVM_K, C, kernel_type, params, 0.5, application)
            )
            minDCFs["gaussianized"][metric].append(
                svm_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, SVM_K, C, kernel_type, params, 0.5, application)
            )

    # Plot Z-NORMALIZED minDCFs
    plot_minDCF(kernel_type.value + ", Z-Normalized features", "C", SVM_C, minDCFs["z-normalized"], True, param_name=param_name, extra="")

    # Plot GAUSSIANIZED minDCFs
    plot_minDCF(kernel_type.value + ", Gaussianized features", "C", SVM_C, minDCFs["gaussianized"], True, param_name=param_name, extra="")

    print(minDCFs)

    print_separator()


def GMM_components_selection(
    DTR_kfold_gaussianized: numpy.ndarray,
    DTR_kfold_z_normalized: numpy.ndarray,
    LTR_kfold: numpy.ndarray,
    DVAL_kfold_gaussianized: numpy.ndarray,
    DVAL_kfold_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
) -> None:
    """
    Evaluate different values of the hyperparameter C for Support Vector Machines by plotting the minDCF
    """

    print_separator("GMM hyperparameter selection")

    minDCFs = {model_type: {"Z-normalization": [], "Gaussianization": []} for model_type in MVGModel}

    for steps in GMM_STEPS:

        # GAUSSIANIZED FEATURES #
        validated_gmm_models = gmm_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, (0.5, 1, 1), steps)
        for model_type, gmm_minDCF in validated_gmm_models.items():
            minDCFs[model_type]["Z-normalization"].append(gmm_minDCF)

        # Z-NORMALIZED FEATURES #
        validated_gmm_models = gmm_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, (0.5, 1, 1), steps)
        for model_type, gmm_minDCF in validated_gmm_models.items():
            minDCFs[model_type]["Gaussianization"].append(gmm_minDCF)

    components = numpy.power(2, numpy.array(GMM_STEPS))
    for model_type in MVGModel:
        if model_type != MVGModel.TIED_NAIVE:
            plot_minDCF(model_type.value, "Components", components, minDCFs[model_type], max_y=0.3)

    print_separator()


####################
# Model evaluation #
####################


def evaluate_MVG_models(
    DTR_kfold_raw: numpy.ndarray,
    DTR_kfold_gaussianized: numpy.ndarray,
    DTR_kfold_z_normalized: numpy.ndarray,
    LTR_kfold: numpy.ndarray,
    DVAL_kfold_raw: numpy.ndarray,
    DVAL_kfold_gaussianized: numpy.ndarray,
    DVAL_kfold_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
) -> None:

    print_separator("MVG MODELS")

    for application in APPLICATIONS:

        # RAW FEATURES #
        validated_mvg_models = mvg_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, LABELS, application)
        print("# RAW Features - {} #".format(application))
        for model_type, model_minDCF in validated_mvg_models.items():
            print("{}: {:.3f}".format(model_type, model_minDCF))

        # Z-NORMALIZED FEATURES #
        validated_mvg_models = mvg_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, LABELS, application)
        print("# Z-NORMALIZED Features - {} #".format(application))
        for model_type, model_minDCF in validated_mvg_models.items():
            print("{}: {:.3f}".format(model_type, model_minDCF))

        # GAUSSIANIZED FEATURES #
        validated_mvg_models = mvg_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, LABELS, application)
        print("# GAUSSIANIZED Features - {} #".format(application))
        for model_type, model_minDCF in validated_mvg_models.items():
            print("{}: {:.3f}".format(model_type, model_minDCF))

    print_separator()


def evaluate_LR_models(
    DTR_kfold_raw: numpy.ndarray,
    DTR_kfold_gaussianized: numpy.ndarray,
    DTR_kfold_z_normalized: numpy.ndarray,
    LTR_kfold: numpy.ndarray,
    DVAL_kfold_raw: numpy.ndarray,
    DVAL_kfold_gaussianized: numpy.ndarray,
    DVAL_kfold_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
):

    print_separator("LR MODELS")

    λ = 1e-6
    quadratic_expansion = [False, True]

    for quadratic in quadratic_expansion:

        for application in APPLICATIONS:

            for πT, _, _ in APPLICATIONS:

                print("# Quadratic = {}, λ = {}, πT = {}, {} #".format(quadratic, λ, πT, application))

                # Raw features #
                lr_minDCF = logistic_regression_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, λ, πT, quadratic, application)
                print("RAW: {:.3f}".format(lr_minDCF))

                # Z-NORMALIZED FEATURES #
                lr_minDCF = logistic_regression_kfold(
                    DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, λ, πT, quadratic, application
                )
                print("Z-NORMALIZED: {:.3f}".format(lr_minDCF))

                # GAUSSIANIZED FEATURES #
                lr_minDCF = logistic_regression_kfold(
                    DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, λ, πT, quadratic, application
                )
                print("GAUSSIANIZED: {:.3f}".format(lr_minDCF))

    print_separator()


def evaluate_SVM_models(
    DTR_kfold_raw: numpy.ndarray,
    DTR_kfold_gaussianized: numpy.ndarray,
    DTR_kfold_z_normalized: numpy.ndarray,
    LTR_kfold: numpy.ndarray,
    DVAL_kfold_raw: numpy.ndarray,
    DVAL_kfold_gaussianized: numpy.ndarray,
    DVAL_kfold_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
):

    print_separator("SVM MODELS")

    svm_params = [
        (1.0, 1e-1, KernelType.NO_KERNEL, None),
        (1.0, 1, KernelType.POLYNOMIAL, (2, 0.1)),
        (1.0, 1, KernelType.POLYNOMIAL, (2, 1)),
        (1.0, 1e2, KernelType.RBF, (0.1)),
    ]

    for K, C, kernel_type, kernel_params in svm_params:

        for application in APPLICATIONS:

            for πT, _, _ in [(None, 1, 1), *APPLICATIONS]:

                print("# {} - πT: {} - K: {} - C: {} - Kernel: {} - Params: {} #".format(application, πT, K, C, kernel_type, kernel_params))

                # RAW FEATURES #
                validated_svm = svm_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, K, C, kernel_type, kernel_params, πT, application)
                print("RAW: {:.3f}".format(validated_svm))

                # GAUSSIANIZED FEATURES #
                validated_svm = svm_kfold(
                    DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, K, C, kernel_type, kernel_params, πT, application
                )
                print("GAUSSIANIZED: {:.3f}".format(validated_svm))

                # Z-NORMALIZED FEATURES #
                validated_svm = svm_kfold(
                    DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, K, C, kernel_type, kernel_params, πT, application
                )
                print("Z-NORMALIZED: {:.3f}".format(validated_svm))

    print_separator()


def evaluate_GMM_models(
    DTR_kfold_raw: numpy.ndarray,
    DTR_kfold_gaussianized: numpy.ndarray,
    DTR_kfold_z_normalized: numpy.ndarray,
    LTR_kfold: numpy.ndarray,
    DVAL_kfold_raw: numpy.ndarray,
    DVAL_kfold_gaussianized: numpy.ndarray,
    DVAL_kfold_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
):

    print_separator("GMM MODELS")

    steps = {
        MVGModel.MVG: 3,
        MVGModel.TIED: 5,
        MVGModel.NAIVE: 5,
        MVGModel.TIED_NAIVE: 5,
    }

    for application in APPLICATIONS:

        print("# GMM - {} #".format(application))

        # RAW FEATURES #
        validated_gmm_models = gmm_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, application, steps)
        print("# RAW Features #")
        for model_type, gmm_minDCF in validated_gmm_models.items():
            print("{} - {} components: {:.3f}".format(model_type, 2 ** steps[model_type], gmm_minDCF))

        # GAUSSIANIZED FEATURES #
        validated_gmm_models = gmm_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, application, steps)
        print("# GAUSSIANIZED Features #")
        for model_type, gmm_minDCF in validated_gmm_models.items():
            print("{} - {} components: {:.3f}".format(model_type, 2 ** steps[model_type], gmm_minDCF))

        # Z-NORMALIZED FEATURES #
        validated_gmm_models = gmm_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, application, steps)
        print("# Z-NORMALIZED Features #")
        for model_type, gmm_minDCF in validated_gmm_models.items():
            print("{} - {} components: {:.3f}".format(model_type, 2 ** steps[model_type], gmm_minDCF))

    print_separator()


####################
# Model Comparison #
####################


def recalibrate_scores(scores: numpy.ndarray, labels: numpy.ndarray, π_tilde: float):
    """Train a LR model to recalibrate scores"""

    DTR_kfold, LTR_kfold, DVAL_kfold = numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray)
    for i, (DTR, LTR, DVAL, _LVAL) in enumerate(kfold(scores, labels, K)):
        DTR_kfold[i], LTR_kfold[i], DVAL_kfold[i] = DTR, LTR, DVAL

    # Train a LR model to recalibrate scores
    recalibrated_scores = logistic_regression_kfold(
        DTR_kfold, LTR_kfold, DVAL_kfold, labels, λ=0, πT=0.5, quadratic=False, application=(π_tilde, 1, 1), return_scores=True
    )
    
    return recalibrated_scores - numpy.log(π_tilde / (1 - π_tilde))


def recalibrate_models(models: dict, labels: numpy.ndarray):
    """Recalibrate models scores"""

    for model, scores in models.items():
        models[model] = recalibrate_scores(numpy.array([scores]), labels, π_tilde=0.5)
        print("{} recalibrated".format(model))
        
    return models


def act_vs_min_DCF(models, LVAL):
    """Draw Bayes error plots"""

    for model, scores in models.items():

        bayes_error_plot(model, scores, LVAL)
