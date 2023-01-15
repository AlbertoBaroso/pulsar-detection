from models.train import mvg_kfold, logistic_regression_kfold
from visualization.evaluation_plots import plot_minDCF_LR
from constants import APPLICATIONS, K, LABELS, LOG_REG_λ
from data_processing.dimensionality_reduction import pca
from visualization.print import print_separator
from data_processing.utils import project_data
from data_processing.validation import kfold
import numpy


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

    Args:
        training_samples (numpy.ndarray): Training samples
        training_labels (numpy.ndarray): Training labels for the samples
    """

    print_separator("LR hyperparameter selection")

    minDCFs = {"z-normalized": {}, "gaussianized": {}}

    for application in APPLICATIONS:

        minDCFs["z-normalized"][application] = []
        minDCFs["gaussianized"][application] = []

        for λ in LOG_REG_λ:

            minDCFs["z-normalized"][application].append(
                logistic_regression_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, λ, 0.5, application)
            )
            minDCFs["gaussianized"][application].append(
                logistic_regression_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, λ, 0.5, application)
            )

    # Plot Z-NORMALIZED minDCFs
    plot_minDCF_LR("Z-Normalized features", minDCFs["z-normalized"])

    # Plot GAUSSIANIZED minDCFs
    plot_minDCF_LR("Gaussianized features", minDCFs["gaussianized"])

    print_separator()


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

    for λ in LOG_REG_λ:

        for application in APPLICATIONS:

            for πT, _, _ in APPLICATIONS:

                print("# LOGISTIC REGRESSION MinDCF, λ = {}, πT = {}, {} #".format(λ, πT, application))

                # Raw features #
                lr_minDCF = logistic_regression_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, λ, πT, application)
                print("RAW: {:.3f}".format(lr_minDCF))

                # Z-NORMALIZED FEATURES #
                lr_minDCF = logistic_regression_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, λ, πT, application)
                print("Z-NORMALIZED: {:.3f}".format(lr_minDCF))

                # GAUSSIANIZED FEATURES #
                lr_minDCF = logistic_regression_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, λ, πT, application)
                print("GAUSSIANIZED: {:.3f}".format(lr_minDCF))
