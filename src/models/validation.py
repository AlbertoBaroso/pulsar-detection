from models.train import mvg_kfold, logistic_regression_kfold, svm_kfold, gmm_kfold, train_log_reg_model, train_gmm_models, train_svm_model
from constants import APPLICATIONS, K, LABELS, LOG_REG_λ, SVM_K, SVM_C, SVM_c, SVM_γ, GMM_STEPS
from visualization.evaluation_plots import plot_minDCF
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

    # Test components = 8, 7, 6, 5, 4, 3
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

            # K-fold cross validation of Gaussian models
            validated_mvg_models = mvg_kfold(DTR_kfold_pca, LTR_kfold, DVAL_kfold_pca, training_labels, LABELS, application)
            print("# PCA m = {} - {} #".format(components, application))
            for model_type, model_minDCF in validated_mvg_models.items():
                print("{}: {:.3f}".format(model_type, model_minDCF))

    print_separator()


def LR_λ_selection(
    DTR_gaussianized: numpy.ndarray,
    DTR_z_normalized: numpy.ndarray,
    LTR: numpy.ndarray,
    DVAL_gaussianized: numpy.ndarray,
    DVAL_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
    quadratic: bool = False,
    use_kfold: bool = True,
    comparison_applications: dict = None,
) -> dict:
    """
    Evaluate different values of the hyperparameter λ for Logistic Regression models by plotting the minDCF
    """

    quadratic_name = "Quadratic " if quadratic else ""
    print_separator("{}LR hyperparameter selection".format(quadratic_name))

    minDCFs = {"z-normalized": {}, "gaussianized": {}}
    if comparison_applications is None:
        z_normalized_comparison, gaussianized_comparison = None, None    
    else:
        z_normalized_comparison, gaussianized_comparison = comparison_applications["z-normalized"], comparison_applications["gaussianized"]

    for application in APPLICATIONS:

        minDCFs["z-normalized"][application] = []
        minDCFs["gaussianized"][application] = []

        for λ in LOG_REG_λ:
            
            if use_kfold:
                minDCF_z_normalized = logistic_regression_kfold(
                    DTR_z_normalized, LTR, DVAL_z_normalized, LVAL, λ, 0.5, quadratic, application
                )
                minDCF_gaussianized = logistic_regression_kfold(
                    DTR_gaussianized, LTR, DVAL_gaussianized, LVAL, λ, 0.5, quadratic, application
                )
            else:
                minDCF_z_normalized = train_log_reg_model(DTR_z_normalized, LTR, DVAL_z_normalized, LVAL, λ, 0.5, quadratic, application)[
                    "minDCF"
                ]
                minDCF_gaussianized = train_log_reg_model(DTR_gaussianized, LTR, DVAL_gaussianized, LVAL, λ, 0.5, quadratic, application)[
                    "minDCF"
                ]

            minDCFs["z-normalized"][application].append(minDCF_z_normalized)
            minDCFs["gaussianized"][application].append(minDCF_gaussianized)

    # Plot Z-NORMALIZED minDCFs
    plot_minDCF(quadratic_name + "LR - Z-Normalized features", "λ", LOG_REG_λ, minDCFs["z-normalized"], True, comparison_applications=z_normalized_comparison)

    # Plot GAUSSIANIZED minDCFs
    plot_minDCF(quadratic_name + "LR - Gaussianized features", "λ", LOG_REG_λ, minDCFs["gaussianized"], True, comparison_applications=gaussianized_comparison)

    print_separator()
    
    return minDCFs


def SVM_C_selection(
    DTR_gaussianized: numpy.ndarray,
    DTR_z_normalized: numpy.ndarray,
    LTR: numpy.ndarray,
    DVAL_gaussianized: numpy.ndarray,
    DVAL_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
    use_kfold: bool = True,
    comparison_applications: dict = None,
) -> dict:
    """
    Evaluate different values of the hyperparameter C for Support Vector Machines by plotting the minDCF
    """

    πTs = [None, 0.5]
    kernel_params = None
    kernel_type = KernelType.NO_KERNEL

    print_separator(kernel_type.value + ", hyperparameter selection")

    minDCFs = {"z-normalized": {}, "gaussianized": {}}
    if comparison_applications is None:
        no_comparison = { πT: None for πT in πTs}
        z_normalized_comparison, gaussianized_comparison = no_comparison, no_comparison    
    else:
        z_normalized_comparison, gaussianized_comparison = comparison_applications["z-normalized"], comparison_applications["gaussianized"]


    for πT in πTs:

        minDCFs["z-normalized"][πT] = {}
        minDCFs["gaussianized"][πT] = {}

        for application in APPLICATIONS:

            minDCFs["z-normalized"][πT][application] = []
            minDCFs["gaussianized"][πT][application] = []

            for C in SVM_C:
                
                if use_kfold:
                    minDCF_z_normalized = svm_kfold(
                        DTR_z_normalized,
                        LTR,
                        DVAL_z_normalized,
                        LVAL,
                        SVM_K,
                        C,
                        kernel_type,
                        kernel_params,
                        πT,
                        application,
                    )
                    minDCF_gaussianized = svm_kfold(
                        DTR_gaussianized,
                        LTR,
                        DVAL_gaussianized,
                        LVAL,
                        SVM_K,
                        C,
                        kernel_type,
                        kernel_params,
                        πT,
                        application,
                    )
                else:
                    minDCF_z_normalized = train_svm_model(DTR_z_normalized, LTR, DVAL_z_normalized, LVAL, SVM_K, C, kernel_type, kernel_params, πT, None, application)["minDCF"]
                    minDCF_gaussianized = train_svm_model(DTR_gaussianized, LTR, DVAL_gaussianized, LVAL, SVM_K, C, kernel_type, kernel_params, πT, None, application)["minDCF"]

                minDCFs["z-normalized"][πT][application].append(minDCF_z_normalized)
                minDCFs["gaussianized"][πT][application].append(minDCF_gaussianized)

    # Plot Z-NORMALIZED minDCFs
    plot_minDCF(kernel_type.value + ", Z-Normalized features, Unbalanced", "C", SVM_C, minDCFs["z-normalized"][πTs[0]], True, comparison_applications=z_normalized_comparison[πTs[0]])
    plot_minDCF(kernel_type.value + ", Z-Normalized features, Balanced", "C", SVM_C, minDCFs["z-normalized"][πTs[1]], True, comparison_applications=z_normalized_comparison[πTs[1]])

    # Plot GAUSSIANIZED minDCFs
    plot_minDCF(kernel_type.value + ", Gaussianized features, Unbalanced", "C", SVM_C, minDCFs["gaussianized"][πTs[0]], True, comparison_applications=gaussianized_comparison[πTs[0]])
    plot_minDCF(kernel_type.value + ", Gaussianized features, Balanced", "C", SVM_C, minDCFs["gaussianized"][πTs[1]], True, comparison_applications=gaussianized_comparison[πTs[1]])

    print_separator()
    
    return minDCFs


def SVM_C_kernel_selection(
    DTR_gaussianized: numpy.ndarray,
    DTR_z_normalized: numpy.ndarray,
    LTR: numpy.ndarray,
    DVAL_gaussianized: numpy.ndarray,
    DVAL_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
    kernel_type: KernelType,
    use_kfold: bool = True,
    comparison_applications: dict = None,
    max_y=0.5
) -> dict:
    """
    Evaluate different values of the hyperparameter C and parameters of the kernel for Support Vector Machines by plotting the minDCF
    """

    print_separator(kernel_type.value + ", hyperparameter selection")

    minDCFs = {"z-normalized": {}, "gaussianized": {}}
    application = APPLICATIONS[0]
    
    if comparison_applications is None:
        z_normalized_comparison, gaussianized_comparison = None, None    
    else:
        z_normalized_comparison, gaussianized_comparison = comparison_applications["z-normalized"], comparison_applications["gaussianized"]

    if kernel_type == KernelType.RBF:
        kernel_params = [(γ) for γ in SVM_γ]
        param_name = r"$\tilde{\pi} = 0.5) - \gamma$"
    elif kernel_type == KernelType.POLYNOMIAL:
        d = 2  # Quadratic polynomial
        kernel_params = [(d, c) for c in SVM_c]
        param_name = r"$\tilde{\pi}$ = 0.5) - c"

    for params in kernel_params:

        metric = params[-1] if type(params) == tuple else params
        minDCFs["z-normalized"][metric] = []
        minDCFs["gaussianized"][metric] = []

        for C in SVM_C:
            
            if use_kfold:
                svm_z_normalized_minDCF = svm_kfold(DTR_z_normalized, LTR, DVAL_z_normalized, LVAL, SVM_K, C, kernel_type, params, 0.5, application)
                svm_gaussianized_minDCF = svm_kfold(DTR_gaussianized, LTR, DVAL_gaussianized, LVAL, SVM_K, C, kernel_type, params, 0.5, application)
            else:
                svm_z_normalized_minDCF = train_svm_model(DTR_z_normalized, LTR, DVAL_z_normalized, LVAL, SVM_K, C, kernel_type, params, 0.5, None, application)["minDCF"]
                svm_gaussianized_minDCF = train_svm_model(DTR_gaussianized, LTR, DVAL_gaussianized, LVAL, SVM_K, C, kernel_type, params, 0.5, None, application)["minDCF"]

            minDCFs["z-normalized"][metric].append(svm_z_normalized_minDCF)
            minDCFs["gaussianized"][metric].append(svm_gaussianized_minDCF)

    # Plot Z-NORMALIZED minDCFs
    plot_minDCF(kernel_type.value + ", Z-Normalized features", "C", SVM_C, minDCFs["z-normalized"], True, param_name=param_name, extra="", max_y=max_y, comparison_applications=z_normalized_comparison)

    # Plot GAUSSIANIZED minDCFs
    plot_minDCF(kernel_type.value + ", Gaussianized features", "C", SVM_C, minDCFs["gaussianized"], True, param_name=param_name, extra="", max_y=max_y, comparison_applications=gaussianized_comparison)

    print_separator()
    
    
    return minDCFs


def GMM_components_selection(
    DTR_gaussianized: numpy.ndarray,
    DTR_z_normalized: numpy.ndarray,
    LTR: numpy.ndarray,
    DVAL_gaussianized: numpy.ndarray,
    DVAL_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
    use_kfold=True,
    comparison_applications: dict = None,
) -> dict:
    """
    Evaluate different values of the hyperparameter C for Support Vector Machines by plotting the minDCF
    """

    print_separator("GMM hyperparameter selection")

    minDCFs = {model_type: {"Z-normalization": [], "Gaussianization": []} for model_type in MVGModel}

    for steps in GMM_STEPS:

        if use_kfold:
            validated_gmm_gaussianized = gmm_kfold(DTR_gaussianized, LTR, DVAL_gaussianized, LVAL, (0.5, 1, 1), steps)
            validated_gmm_z_normalized = gmm_kfold(DTR_z_normalized, LTR, DVAL_z_normalized, LVAL, (0.5, 1, 1), steps)
        else:
            gaussianized_gmm = train_gmm_models(DTR_gaussianized, LTR, DVAL_gaussianized, LVAL, steps)
            z_normalized_gmm = train_gmm_models(DTR_z_normalized, LTR, DVAL_z_normalized, LVAL, steps)
            validated_gmm_gaussianized = {model: values["minDCF"][0.5] for model, values in gaussianized_gmm.items()}
            validated_gmm_z_normalized = {model: values["minDCF"][0.5] for model, values in z_normalized_gmm.items()}

        # GAUSSIANIZED FEATURES #
        for model_type, gmm_minDCF in validated_gmm_gaussianized.items():
            minDCFs[model_type]["Gaussianization"].append(gmm_minDCF)

        # Z-NORMALIZED FEATURES #
        for model_type, gmm_minDCF in validated_gmm_z_normalized.items():
            minDCFs[model_type]["Z-normalization"].append(gmm_minDCF)

    components = numpy.power(2, numpy.array(GMM_STEPS))
    for model_type in MVGModel:
        if model_type != MVGModel.TIED_NAIVE:
            comparison = None if comparison_applications is None else comparison_applications[model_type] 
            plot_minDCF(model_type.value, "Components", components, minDCFs[model_type], max_y=0.3, comparison_applications=comparison)

    print_separator()
    
    return minDCFs


####################
# Model Validation #
####################


def validate_MVG_models(
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


def validate_LR_models(
    DTR_kfold_raw: numpy.ndarray,
    DTR_kfold_gaussianized: numpy.ndarray,
    DTR_kfold_z_normalized: numpy.ndarray,
    LTR_kfold: numpy.ndarray,
    DVAL_kfold_raw: numpy.ndarray,
    DVAL_kfold_gaussianized: numpy.ndarray,
    DVAL_kfold_z_normalized: numpy.ndarray,
    LVAL: numpy.ndarray,
    quadratic: bool = False,
):

    print_separator("LR MODELS")

    λ = 1e-6

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


def validate_SVM_models(
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


def validate_GMM_models(
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
