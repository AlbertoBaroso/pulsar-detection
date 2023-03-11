from models.evaluation import evaluate_mvg, evaluate_lr, evaluate_svm, evaluate_gmm
from data_processing.data_load import load_training, load_test, load_preprocessed
from data_processing.calibration import act_vs_min_DCF, recalibrate_models
from visualization.evaluation_plots import DET_plot
from data_processing.utils import shuffle_data
from models.train import train_best_models
from models.svm import KernelType
from models.validation import (
    PCA_m_selection,
    validate_MVG_models,
    validate_LR_models,
    LR_λ_selection,
    SVM_C_selection,
    SVM_C_kernel_selection,
    validate_SVM_models,
    GMM_components_selection,
    validate_GMM_models,
)

if __name__ == "__main__":

    # Load data from file #

    training_samples, training_labels = load_training()
    test_samples, test_labels = load_test()

    # Shuffle data #
    training_samples, training_labels = shuffle_data(training_samples, training_labels)

    # Pre-process data #
    (
        DTR_gaussianized,
        DTE_gaussianized,
        DTR_kfold_raw,
        DVAL_kfold_raw,
        DTR_z_normalized,
        DTE_z_normalized,
        DTR_kfold_gaussianized,
        DVAL_kfold_gaussianized,
        DTR_kfold_z_normalized,
        DVAL_kfold_z_normalized,
        LTR_kfold,
        LVAL_kfold,
    ) = load_preprocessed(training_samples, training_labels, test_samples)

    LVAL = training_labels
    LTR = training_labels
    LTE = test_labels

    #################################
    # PCA hyper parameter selection #
    #################################

    PCA_m_selection(training_samples, training_labels)

    ##############
    # MVG Models #
    ##############

    validate_MVG_models(
        DTR_kfold_raw,
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_raw,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL,
    )

    #######################
    # Logistic Regression #
    #######################

    # Hyperparameter selection
    LR_λ_minDCF = LR_λ_selection(
        DTR_kfold_gaussianized, DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_gaussianized, DVAL_kfold_z_normalized, LVAL
    )

    # Model evaluation
    validate_LR_models(
        DTR_kfold_raw,
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_raw,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL,
    )

    ##############
    # SVM MODELS #
    ##############

    # Hyperparameter selection: Linear SVM
    linear_svm_minDCF = SVM_C_selection(
        DTR_kfold_gaussianized, DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_gaussianized, DVAL_kfold_z_normalized, LVAL
    )

    # Hyperparameter selection: Polynomial kernel SVM
    polynomial_svm_minDCF = SVM_C_kernel_selection(
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL,
        KernelType.POLYNOMIAL,
    )

    # Hyperparameter selection: RBF kernel SVM
    rbf_svm_minDCF = SVM_C_kernel_selection(
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL,
        KernelType.RBF,
        use_kfold=True,
    )

    # Model evaluation
    validate_SVM_models(
        DTR_kfold_raw,
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_raw,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL,
    )

    ##############
    # GMM MODELS #
    ##############

    # Hyperparameter selection
    gmm_minDCF = GMM_components_selection(
        DTR_kfold_gaussianized, DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_gaussianized, DVAL_kfold_z_normalized, LVAL
    )

    # Model evaluation
    validate_GMM_models(
        DTR_kfold_raw,
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_raw,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL,
    )

    ####################
    # Model Comparison #
    ####################

    best_models_validation = train_best_models(
        DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, DTR_kfold_gaussianized, DVAL_kfold_gaussianized, use_kfold=True
    )

    act_vs_min_DCF(best_models_validation, LVAL)

    DET_plot(best_models_validation, LVAL)

    # Score recalibration

    best_models_validation_recalibrated = recalibrate_models(best_models_validation, LVAL, kfold=True)

    act_vs_min_DCF(best_models_validation_recalibrated, LVAL)

    # Use recalibrated models on test data

    recalibration_parameters = recalibrate_models(best_models_validation, LVAL)

    best_models_full = train_best_models(
        DTR_z_normalized,
        LTR,
        DTE_z_normalized,
        LTE,
        DTR_gaussianized,
        DTE_gaussianized,
        use_kfold=False,
        recalibration_parameters=recalibration_parameters,
    )

    for model_name, values in best_models_full.items():
        print(model_name)
        print("MinDCF: ", values["minDCF"])
        print("ActDCF: ", values["actDCF"])
        print("Error Rate: ", values["error_rate"])

    ########################
    # Experimental results #
    ########################

    data = {
        "Raw features": (training_samples, test_samples),
        "Z-normalized features": (DTR_z_normalized, DTE_z_normalized),
        "Gaussianized features": (DTR_gaussianized, DTE_gaussianized),
    }

    # Multivariate Gaussian classifiers

    evaluate_mvg(data, LTR, LTE)

    # Linear Logistic Regression

    LR_λ_selection(
        DTR_gaussianized,
        DTR_z_normalized,
        LTR,
        DTE_gaussianized,
        DTE_z_normalized,
        LTE,
        quadratic=False,
        use_kfold=False,
        comparison_applications=LR_λ_minDCF,
    )

    evaluate_lr(data, LTR, LTE, quadratic=False)

    # Linear SVM

    SVM_C_selection(
        DTR_gaussianized,
        DTR_z_normalized,
        LTR,
        DTE_gaussianized,
        DTE_z_normalized,
        LTE,
        use_kfold=False,
        comparison_applications=linear_svm_minDCF,
    )

    evaluate_svm(data, LTR, LTE, KernelType.NO_KERNEL, ())

    # Polynomial Kernel SVM

    SVM_C_kernel_selection(
        DTR_gaussianized,
        DTR_z_normalized,
        LTR,
        DTE_gaussianized,
        DTE_z_normalized,
        LTE,
        kernel_type=KernelType.POLYNOMIAL,
        use_kfold=False,
        comparison_applications=polynomial_svm_minDCF,
        max_y=0.4,
    )

    evaluate_svm(data, LTR, LTE, KernelType.POLYNOMIAL, (2, 0.1))

    # RBF SVM

    SVM_C_kernel_selection(
        DTR_gaussianized,
        DTR_z_normalized,
        LTR,
        DTE_gaussianized,
        DTE_z_normalized,
        LTE,
        kernel_type=KernelType.RBF,
        use_kfold=False,
        comparison_applications=rbf_svm_minDCF,
        max_y=0.25,
    )

    evaluate_svm(data, LTR, LTE, KernelType.RBF, (0.1))

    # GMM

    GMM_components_selection(
        DTR_gaussianized,
        DTR_z_normalized,
        LTR,
        DTE_gaussianized,
        DTE_z_normalized,
        LTE,
        use_kfold=False,
        comparison_applications=gmm_minDCF,
    )

    evaluate_gmm(
        data,
        LTR,
        LTE,
    )
