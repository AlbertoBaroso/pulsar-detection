from data_processing.data_load import load_training, load_test, load_preprocessed
from models.train import mvg_kfold, logistic_regression_kfold, gmm_kfold, svm_kfold
from constants import K, LABELS, APPLICATIONS, LOG_REG_λ
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
        DTR_pca,
        DTE_pca,
        DTR_lda,
        DTE_lda,
        DTR_kfold_raw,
        DVAL_kfold_raw,
        DTR_z_normalized,
        DTE_z_normalized,
        DTR_kfold_gaussianized,
        DVAL_kfold_gaussianized,
        DTR_kfold_z_normalized,
        DVAL_kfold_z_normalized,
        DTR_kfold_pca,
        DVAL_kfold_pca,
        DTR_kfold_lda,
        DVAL_kfold_lda,
        LTR_kfold,
        LVAL_kfold,
    ) = load_preprocessed(training_samples, training_labels, test_samples, test_labels)
    LVAL = training_labels

    #######################
    # Logistic Regression #
    #######################

    # for λ in LOG_REG_λ:

    #     for application in APPLICATIONS:
        
    #         for πT, _, _ in APPLICATIONS:

    #             print("# LOGISTIC REGRESSION MinDCF, λ = {}, πT = {}, {} #".format(λ, πT, application))
            
    #             # Raw features #
    #             logisitc_regression_error = logistic_regression_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, λ,, πT application)
    #             print("RAW: {}".format(logisitc_regression_error))

    #             # PCA #
    #             logisitc_regression_error = logistic_regression_kfold(DTR_kfold_pca, LTR_kfold, DVAL_kfold_pca, LVAL, λ, πT, application)
    #             print("PCA: {}".format(logisitc_regression_error))

    #             # LDA #
    #             logisitc_regression_error = logistic_regression_kfold(DTR_kfold_lda, LTR_kfold, DVAL_kfold_lda, LVAL, λ, πT, application)
    #             print("LDA: {}".format(logisitc_regression_error))

    #             # GAUSSIANIZED FEATURES #
    #             logisitc_regression_error = logistic_regression_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, λ, πT, application)
    #             print("GAUSSIANIZED: {}".format(logisitc_regression_error))
                
    #             # Z-NORMALIZED FEATURES #
    #             logisitc_regression_error = logistic_regression_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, λ, πT, application)
    #             print("Z-NORMALIZED: {}".format(logisitc_regression_error))
        
    # # print("\n" * 5)

    # ##############
    # # MVG MODELS #
    # ##############

    # for application in APPLICATIONS:

    #     # RAW FEATURES #
    #     validated_mvg_models = mvg_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, LABELS, application)
    #     print("# RAW Features - {} #".format(application))
    #     for model_type, model_minDCF in validated_mvg_models.items():
    #         print("{}: {}".format(model_type, model_minDCF))

    #     # LDA #
    #     validated_mvg_models = mvg_kfold(DTR_kfold_lda, LTR_kfold, DVAL_kfold_lda, LVAL, LABELS, application)
    #     print("# LDA Features - {} #".format(application))
    #     for model_type, model_minDCF in validated_mvg_models.items():
    #         print("{}: {}".format(model_type, model_minDCF))

    #     # PCA #
    #     validated_mvg_models = mvg_kfold(DTR_kfold_pca, LTR_kfold, DVAL_kfold_pca, LVAL, LABELS, application)
    #     print("# PCA Features - {} #".format(application))
    #     for model_type, model_minDCF in validated_mvg_models.items():
    #         print("{}: {}".format(model_type, model_minDCF))

    #     # GAUSSIANIZED FEATURES #
    #     validated_mvg_models = mvg_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, LABELS, application)
    #     print("# GAUSSIANIZED Features - {} #".format(application))
    #     for model_type, model_minDCF in validated_mvg_models.items():
    #         print("{}: {}".format(model_type, model_minDCF))

    #     # Z-NORMALIZED FEATURES #
    #     validated_mvg_models = mvg_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, LABELS, application)
    #     print("# Z-NORMALIZED Features - {} #".format(application))
    #     for model_type, model_minDCF in validated_mvg_models.items():
    #         print("{}: {}".format(model_type, model_minDCF))

    # ##############
    # # GMM MODELS #
    # ##############
    
    # class_priors = [0.5, 0.5]
    
    # for steps in range(1, 5):
        
    #     # RAW FEATURES #
    #     validated_gmm_models = gmm_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, class_priors, steps)
    #     print("# RAW Features - Priors: {} - Steps: {} #".format(class_priors, steps))
    #     for model_type, model_error_rate in validated_gmm_models.items():
    #         print("{}: {}".format(model_type, model_error_rate))
            
    #     # LDA #
    #     validated_gmm_models = gmm_kfold(DTR_kfold_lda, LTR_kfold, DVAL_kfold_lda, LVAL, class_priors, steps)
    #     print("# LDA Features - Priors: {} - Steps: {} #".format(class_priors, steps))
    #     for model_type, model_error_rate in validated_gmm_models.items():
    #         print("{}: {}".format(model_type, model_error_rate))

    #     # PCA #
    #     validated_gmm_models = gmm_kfold(DTR_kfold_pca, LTR_kfold, DVAL_kfold_pca, LVAL, class_priors, steps)
    #     print("# PCA Features - Priors: {} - Steps: {} #".format(class_priors, steps))
    #     for model_type, model_error_rate in validated_gmm_models.items():
    #         print("{}: {}".format(model_type, model_error_rate))

    #     # GAUSSIANIZED FEATURES #
    #     validated_gmm_models = gmm_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, class_priors, steps)
    #     print("# GAUSSIANIZED Features - Priors: {} - Steps: {} #".format(class_priors, steps))
    #     for model_type, model_error_rate in validated_gmm_models.items():
    #         print("{}: {}".format(model_type, model_error_rate))

    #     # Z-NORMALIZED FEATURES #
    #     validated_gmm_models = gmm_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, class_priors, steps)
    #     print("# Z-NORMALIZED Features - Priors: {} - Steps: {} #".format(class_priors, steps))
    #     for model_type, model_error_rate in validated_gmm_models.items():
    #         print("{}: {}".format(model_type, model_error_rate))
    
    ##############
    # SVM MODELS #
    ##############

    svm_params = [
        (1.0, 0.01, KernelType.NO_KERNEL, None),
        # (1.0, 0.1, KernelType.NO_KERNEL, None),
        # (1.0, 1.0, KernelType.NO_KERNEL, None),
        # (1.0, 10.0, KernelType.NO_KERNEL, None),
        # (10.0, 0.1, KernelType.NO_KERNEL, None),
        # (10.0, 1.0, KernelType.NO_KERNEL, None),
        # (10.0, 10, KernelType.NO_KERNEL, None),
        # (0.0, 1.0, KernelType.POLYNOMIAL, (2, 0)),
        # (1.0, 1.0, KernelType.POLYNOMIAL, (2, 0)),
        # (0.0, 1.0, KernelType.POLYNOMIAL, (2, 1)),
        # (1.0, 1.0, KernelType.POLYNOMIAL, (2, 1)),
        # (0.0, 1.0, KernelType.RBF, (1.0)),
        # (0.0, 1.0, KernelType.RBF, (10.0)),
        # (1.0, 1.0, KernelType.RBF, (1.0)),
        # (1.0, 1.0, KernelType.RBF, (10.0)),
    ]
    
    for K, C, kernel_type, kernel_params in svm_params:
    
        for application in APPLICATIONS:
        
            for πT, _, _ in [(None, 1, 1), *APPLICATIONS]:
        
                print("# RAW Features - {}, πT: {} - K: {} - C: {} - Kernel: {} - Params: {} #".format(application, πT, K, C, kernel_type, kernel_params))
                
                # # RAW FEATURES #
                # validated_svm = svm_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, K, C, kernel_type, kernel_params, πT, application)
                # print("RAW: {}".format(validated_svm))
                    
                # # LDA #
                # validated_svm = svm_kfold(DTR_kfold_lda, LTR_kfold, DVAL_kfold_lda, LVAL, K, C, kernel_type, kernel_params, πT, application)
                # print("LDA: {}".format(validated_svm))

                # # PCA #
                # validated_svm = svm_kfold(DTR_kfold_pca, LTR_kfold, DVAL_kfold_pca, LVAL, K, C, kernel_type, kernel_params, πT, application)
                # print("PCA: {}".format(validated_svm))

                # # GAUSSIANIZED FEATURES #
                # validated_svm = svm_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, K, C, kernel_type, kernel_params, πT, application)
                # print("GAUSSIANIZED: {}".format(validated_svm))

                # Z-NORMALIZED FEATURES #
                validated_svm = svm_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, K, C, kernel_type, kernel_params, πT, application)
                print("Z-NORMALIZED: {}".format(validated_svm))