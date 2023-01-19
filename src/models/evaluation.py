from models.train import mvg_kfold, logistic_regression_kfold, svm_kfold, gmm_kfold
from constants import APPLICATIONS, K, LABELS, LOG_REG_λ, SVM_K, SVM_C, SVM_γ, GMM_STEPS
from data_processing.dimensionality_reduction import pca
from visualization.evaluation_plots import plot_minDCF
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
    kernel_type: KernelType = KernelType.NO_KERNEL
) -> None:
    """
    Evaluate different values of the hyperparameter C for Support Vector Machines by plotting the minDCF
    """

    πTs = [None, 0.5]
        
    if kernel_type == KernelType.RBF:
        kernel_params = (0.5)
    elif kernel_type == KernelType.POLYNOMIAL:
        kernel_params = (2, 0)
    else:
        kernel_params = None
    
    import datetime

    print_separator(kernel_type.value + ", hyperparameter selection")

    minDCFs = {"z-normalized": {}, "gaussianized": {}}

    for πT in πTs:

        minDCFs["z-normalized"][πT] = {}
        minDCFs["gaussianized"][πT] = {}

        for application in APPLICATIONS:
            
            print(">>>>> πT = {} - {} <<<<<".format(πT, application))

            minDCFs["z-normalized"][πT][application] = []
            minDCFs["gaussianized"][πT][application] = []

            for C in SVM_C:
                
                print("{}, ".format(C), end="", flush=True)

                minDCFs["z-normalized"][πT][application].append(
                    svm_kfold(
                        DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, SVM_K, C, kernel_type, kernel_params, πT, application
                    )
                )
                minDCFs["gaussianized"][πT][application].append(
                    svm_kfold(
                        DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, SVM_K, C, kernel_type, kernel_params, πT, application
                    )
                )
                
                print("<< {} >>".format(datetime.datetime.now()))

    # Plot Z-NORMALIZED minDCFs
    print("DCFS: ", minDCFs["z-normalized"][πTs[0]])
    plot_minDCF(kernel_type.value + ", Z-Normalized features, Unbalanced", "C", SVM_C, minDCFs["z-normalized"][πTs[0]], True)
    plot_minDCF(kernel_type.value + ", Z-Normalized features, Balanced", "C", SVM_C, minDCFs["z-normalized"][πTs[1]], True)

    # Plot GAUSSIANIZED minDCFs
    plot_minDCF(kernel_type.value + ", Gaussianized features, Unbalanced", "C", SVM_C, minDCFs["gaussianized"][πTs[0]], True)
    plot_minDCF(kernel_type.value + ", Gaussianized features, Balanced", "C", SVM_C, minDCFs["gaussianized"][πTs[1]], True)

    print_separator()


def SVM_C_γ_selection(
    DTR_kfold_gaussianized: numpy.ndarray,
    DTR_kfold_z_normalized: numpy.ndarray,
    LTR_kfold: numpy.ndarray,
    DVAL_kfold_gaussianized: numpy.ndarray,
    DVAL_kfold_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray
) -> None:
    """
    Evaluate different values of the hyperparameters C and γ for Support Vector Machines with RBF Kernel by plotting the minDCF
    """

    import datetime

    print_separator("RBF Kernel SVM, hyperparameter selection")

    minDCFs = {"z-normalized": {}, "gaussianized": {}}
    application = APPLICATIONS[0]

    for γ in SVM_γ: 
        
        minDCFs["z-normalized"][γ] = []
        minDCFs["gaussianized"][γ] = []
        
        for C in SVM_C:
            
            print("C={}, γ={} ".format(C, γ), end="", flush=True)

            minDCFs["z-normalized"][γ].append(
                svm_kfold(
                    DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, SVM_K, C, KernelType.RBF, (γ), None, application
                )
            )
            minDCFs["gaussianized"][γ].append(
                svm_kfold(
                    DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, SVM_K, C, KernelType.RBF, (γ), None, application
                )
            )
            
            print("[{}, {}] << {} >>".format(minDCFs["z-normalized"][γ], minDCFs["gaussianized"][γ], datetime.datetime.now()))

    # Plot Z-NORMALIZED minDCFs
    print("DCFS: ", minDCFs["z-normalized"])
    plot_minDCF(KernelType.RBF.value + ", Z-Normalized features, Unbalanced", "C", SVM_C, minDCFs["z-normalized"], True, param_name='$\lambda$', extra='')

    # # Plot GAUSSIANIZED minDCFs
    plot_minDCF(KernelType.RBF.value + ", Gaussianized features, Unbalanced", "C", SVM_C, minDCFs["gaussianized"], True, param_name='$\lambda$', extra='')

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

    minDCFs = {
        model_type: {"Z-normalization": [], "Gaussianization": []}
        for model_type in MVGModel
    }
    
    import datetime
    
    for steps in GMM_STEPS:
        
        # GAUSSIANIZED FEATURES #
        validated_gmm_models = gmm_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, (0.5, 1, 1), steps)
        for model_type, gmm_minDCF in validated_gmm_models.items():
            minDCFs[model_type]["Z-normalization"].append(gmm_minDCF)

        # Z-NORMALIZED FEATURES #
        validated_gmm_models = gmm_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, (0.5, 1, 1), steps)
        for model_type, gmm_minDCF in validated_gmm_models.items():
            minDCFs[model_type]["Gaussianization"].append(gmm_minDCF)
            
        print("{} Components finished at: {}".format(2**steps, datetime.datetime.now()))
            

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
        (1.0, 1e-2, KernelType.NO_KERNEL, None),
        # (1.0, 1e-2, KernelType.POLYNOMIAL, (2, 0)),
        # (0.0, 1e-2, KernelType.RBF, (1.0)),
        # (0.0, 1e-2, KernelType.RBF, (10.0)),
        # (1.0, 1e-2, KernelType.RBF, (1.0)),
        # (1.0, 1e-2, KernelType.RBF, (10.0)),
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

    steps = 10  # TODO

    for application in APPLICATIONS:

        print("# GMM - {} - {} components #".format(application, 2**steps))

        # RAW FEATURES #
        validated_gmm_models = gmm_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, application, steps)
        print("# RAW Features #")
        for model_type, gmm_minDCF in validated_gmm_models.items():
            print("{}: {:.3f}".format(model_type, gmm_minDCF))

        # GAUSSIANIZED FEATURES #
        validated_gmm_models = gmm_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, application, steps)
        print("# GAUSSIANIZED Features #")
        for model_type, gmm_minDCF in validated_gmm_models.items():
            print("{}: {:.3f}".format(model_type, gmm_minDCF))

        # Z-NORMALIZED FEATURES #
        validated_gmm_models = gmm_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, application, steps)
        print("# Z-NORMALIZED Features #")
        for model_type, gmm_minDCF in validated_gmm_models.items():
            print("{}: {:.3f}".format(model_type, gmm_minDCF))

    print_separator()
