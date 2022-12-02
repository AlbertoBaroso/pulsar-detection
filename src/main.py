from data_processing.data_load import load_training, load_test, load_preprocessed
from models.train import mvg_kfold, train_log_reg_model
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

    #######################
    # Logistic Regression #
    #######################

    log_reg_raw_results = []
    log_reg_pca_results = []
    log_reg_lda_results = []
    log_reg_gaussianized_results = []

    for λ in LOG_REG_λ:

        # Raw features #
        logisitc_regression_error = train_log_reg_model(training_samples, training_labels, test_samples, test_labels, λ)
        log_reg_raw_results.append((λ, logisitc_regression_error))

        # PCA #
        logisitc_regression_error = train_log_reg_model(DTR_pca, training_labels, DTE_pca, test_labels, λ)
        log_reg_pca_results.append((λ, logisitc_regression_error))

        # LDA #
        logisitc_regression_error = train_log_reg_model(DTR_lda, training_labels, DTE_lda, test_labels, λ)
        log_reg_lda_results.append((λ, logisitc_regression_error))

        # GAUSSIANIZED FEATURES #
        logisitc_regression_error = train_log_reg_model(DTR_gaussianized, training_labels, DTE_gaussianized, test_labels, λ)
        log_reg_gaussianized_results.append((λ, logisitc_regression_error))

    print("\n" * 5)
    print("# LOGISTIC REGRESSION ERROR RATES #")
    for λ, logisitc_regression_error in log_reg_raw_results:
        print("LogReg RAW with λ = {} : {}".format(λ, logisitc_regression_error))
    for λ, logisitc_regression_error in log_reg_pca_results:
        print("LogReg PCA with λ = {} : {}".format(λ, logisitc_regression_error))
    for λ, logisitc_regression_error in log_reg_lda_results:
        print("LogReg LDA with λ = {} : {}".format(λ, logisitc_regression_error))
    for λ, logisitc_regression_error in log_reg_gaussianized_results:
        print("LogReg GAUSSIANIZED with λ = {} : {}".format(λ, logisitc_regression_error))

    ##############
    # MVG MODELS #
    ##############

    for class_priors in CLASS_PRIORS:

        # RAW FEATURES #
        validated_mvg_models = mvg_kfold(DTR_kfold_raw, LTR_kfold, DVAL_kfold_raw, training_labels, class_priors, labels)
        print("# RAW Features - Priors: {} #".format(class_priors))
        for model_type, model_error_rate in validated_mvg_models.items():
            print("{}: {}".format(model_type, model_error_rate))

        # LDA #
        validated_mvg_models = mvg_kfold(DTR_kfold_lda, LTR_kfold, DVAL_kfold_lda, training_labels, class_priors, labels)
        print("# LDA Features - Priors: {} #".format(class_priors))
        for model_type, model_error_rate in validated_mvg_models.items():
            print("{}: {}".format(model_type, model_error_rate))

        # PCA #
        validated_mvg_models = mvg_kfold(DTR_kfold_pca, LTR_kfold, DVAL_kfold_pca, training_labels, class_priors, labels)
        print("# PCA Features - Priors: {} #".format(class_priors))
        for model_type, model_error_rate in validated_mvg_models.items():
            print("{}: {}".format(model_type, model_error_rate))

        # GAUSSIANIZED FEATURES #
        validated_mvg_models = mvg_kfold(DTR_kfold_gaussianized, LTR_kfold, DVAL_kfold_gaussianized, training_labels, class_priors, labels)
        print("# GAUSSIANIZED Features - Priors: {} #".format(class_priors))
        for model_type, model_error_rate in validated_mvg_models.items():
            print("{}: {}".format(model_type, model_error_rate))

        # Z-NORMALIZED FEATURES #
        validated_mvg_models = mvg_kfold(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, training_labels, class_priors, labels)
        print("# Z-NORMALIZED Features - Priors: {} #".format(class_priors))
        for model_type, model_error_rate in validated_mvg_models.items():
            print("{}: {}".format(model_type, model_error_rate))
