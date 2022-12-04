from data_processing.data_load import load_training, load_test, load_preprocessed
from models.train import mvg_kfold, logistic_regression_kfold
from constants import K, labels, CLASS_PRIORS, LOG_REG_λ

if __name__ == "__main__":

    # Load data from file #

    training_samples, training_labels = load_training()
    test_samples, test_labels = load_test()

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

    for λ in LOG_REG_λ:

        print("# LOGISTIC REGRESSION ERROR RATES, λ = {} #".format(λ))
    
        # Raw features #
        logisitc_regression_error = logistic_regression_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, λ)
        print("RAW: {}".format(logisitc_regression_error))

        # PCA #
        logisitc_regression_error = logistic_regression_kfold(DTR_kfold_pca, LTR_kfold, DVAL_kfold_pca, LVAL, λ)
        print("PCA: {}".format(logisitc_regression_error))

        # LDA #
        logisitc_regression_error = logistic_regression_kfold(DTR_kfold_lda, LTR_kfold, DVAL_kfold_lda, LVAL, λ)
        print("LDA: {}".format(logisitc_regression_error))

        # GAUSSIANIZED FEATURES #
        logisitc_regression_error = logistic_regression_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, λ)
        print("GAUSSIANIZED: {}".format(logisitc_regression_error))
        
        # Z-NORMALIZED FEATURES #
        logisitc_regression_error = logistic_regression_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, λ)
        print("Z-NORMALIZED: {}".format(logisitc_regression_error))
        
    print("\n" * 5)

    ##############
    # MVG MODELS #
    ##############

    for class_priors in CLASS_PRIORS:

        # RAW FEATURES #
        validated_mvg_models = mvg_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, LVAL, class_priors, labels)
        print("# RAW Features - Priors: {} #".format(class_priors))
        for model_type, model_error_rate in validated_mvg_models.items():
            print("{}: {}".format(model_type, model_error_rate))

        # LDA #
        validated_mvg_models = mvg_kfold(DTR_kfold_lda, LTR_kfold, DVAL_kfold_lda, LVAL, class_priors, labels)
        print("# LDA Features - Priors: {} #".format(class_priors))
        for model_type, model_error_rate in validated_mvg_models.items():
            print("{}: {}".format(model_type, model_error_rate))

        # PCA #
        validated_mvg_models = mvg_kfold(DTR_kfold_pca, LTR_kfold, DVAL_kfold_pca, LVAL, class_priors, labels)
        print("# PCA Features - Priors: {} #".format(class_priors))
        for model_type, model_error_rate in validated_mvg_models.items():
            print("{}: {}".format(model_type, model_error_rate))

        # GAUSSIANIZED FEATURES #
        validated_mvg_models = mvg_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, LVAL, class_priors, labels)
        print("# GAUSSIANIZED Features - Priors: {} #".format(class_priors))
        for model_type, model_error_rate in validated_mvg_models.items():
            print("{}: {}".format(model_type, model_error_rate))

        # Z-NORMALIZED FEATURES #
        validated_mvg_models = mvg_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL, class_priors, labels)
        print("# Z-NORMALIZED Features - Priors: {} #".format(class_priors))
        for model_type, model_error_rate in validated_mvg_models.items():
            print("{}: {}".format(model_type, model_error_rate))
