from models.evaluation import PCA_m_selection, evaluate_MVG_models, evaluate_LR_models, LR_λ_selection
from models.train import logistic_regression_kfold, gmm_kfold, svm_kfold
from data_processing.data_load import load_training, load_test, load_preprocessed
from constants import K, APPLICATIONS
from data_processing.utils import shuffle_data
from models.svm import KernelType

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
    ) = load_preprocessed(training_samples, training_labels, test_samples, test_labels)
    LVAL = training_labels

    #################################
    # PCA hyper parameter selection #
    #################################

    # PCA_m_selection(training_samples, training_labels)

    ##############
    # MVG Models #
    ##############

    # MVG MODELS
    # evaluate_MVG_models(
    #     DTR_kfold_raw,
    #     DTR_kfold_gaussianized,
    #     DTR_kfold_z_normalized,
    #     LTR_kfold,
    #     DVAL_kfold_raw,
    #     DVAL_kfold_gaussianized,
    #     DVAL_kfold_z_normalized,
    #     LVAL
    # )

    #######################
    # Logistic Regression #
    #######################
    
    # Hyperparameter selection
    LR_λ_selection(
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL
    )

    # Model evaluation
    evaluate_LR_models(
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

    # for application in APPLICATIONS:

    #     for steps in range(1, 8):

    #         print("# GMM - {} - {} components #".format(application, 2**steps))

    #         # RAW FEATURES #
    #         validated_gmm_models = gmm_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, application, steps)
    #         print("# RAW Features #")
    #         for model_type, gmm_minDCF in validated_gmm_models.items():
    #             print("{}: {:.3f}".format(model_type, gmm_minDCF))

    #         # GAUSSIANIZED FEATURES #
    #         validated_gmm_models = gmm_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, application, steps)
    #         print("# GAUSSIANIZED Features #")
    #         for model_type, gmm_minDCF in validated_gmm_models.items():
    #             print("{}: {:.3f}".format(model_type, gmm_minDCF))

    #         # Z-NORMALIZED FEATURES #
    #         validated_gmm_models = gmm_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, application, steps)
    #         print("# Z-NORMALIZED Features #")
    #         for model_type, gmm_minDCF in validated_gmm_models.items():
    #             print("{}: {:.3f}".format(model_type, gmm_minDCF))

    ##############
    # SVM MODELS #
    ##############

    # svm_params = [
    #     (1.0, 0.1, KernelType.NO_KERNEL, None),
    #     (1.0, 1.0, KernelType.NO_KERNEL, None),
    #     (1.0, 10.0, KernelType.NO_KERNEL, None),
    #     (10.0, 0.1, KernelType.NO_KERNEL, None),
    #     (10.0, 1.0, KernelType.NO_KERNEL, None),
    #     (10.0, 10, KernelType.NO_KERNEL, None),
    #     (0.0, 1.0, KernelType.POLYNOMIAL, (2, 0)),
    #     (1.0, 1.0, KernelType.POLYNOMIAL, (2, 0)),
    #     (0.0, 1.0, KernelType.POLYNOMIAL, (2, 1)),
    #     (1.0, 1.0, KernelType.POLYNOMIAL, (2, 1)),
    #     (0.0, 1.0, KernelType.RBF, (1.0)),
    #     (0.0, 1.0, KernelType.RBF, (10.0)),
    #     (1.0, 1.0, KernelType.RBF, (1.0)),
    #     (1.0, 1.0, KernelType.RBF, (10.0)),
    # ]

    # for K, C, kernel_type, kernel_params in svm_params:

    #     for application in APPLICATIONS:

    #         for πT, _, _ in [(None, 1, 1), *APPLICATIONS]:

    #             print("# {}, πT: {} - K: {} - C: {} - Kernel: {} - Params: {} #".format(application, πT, K, C, kernel_type, kernel_params))

    #             # RAW FEATURES #
    #             validated_svm = svm_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, K, C, kernel_type, kernel_params, πT, application)
    #             print("RAW: {}".format(validated_svm))

    #             # GAUSSIANIZED FEATURES #
    #             validated_svm = svm_kfold(
    #                 DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, K, C, kernel_type, kernel_params, πT, application
    #             )
    #             print("GAUSSIANIZED: {:.3f}".format(validated_svm))

    #             # Z-NORMALIZED FEATURES #
    #             validated_svm = svm_kfold(
    #                 DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, K, C, kernel_type, kernel_params, πT, application
    #             )
    #             print("Z-NORMALIZED: {:.3f}".format(validated_svm))
