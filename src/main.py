from models.evaluation import PCA_m_selection, evaluate_MVG_models, evaluate_LR_models, LR_λ_selection, SVM_C_selection, SVM_C_kernel_selection, evaluate_SVM_models, GMM_components_selection, evaluate_GMM_models, act_vs_min_DCF, recalibrate_models
from data_processing.data_load import load_training, load_test, load_preprocessed
from data_processing.utils import shuffle_data
from models.train import train_best_models
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

    PCA_m_selection(training_samples, training_labels)

    ##############
    # MVG Models #
    ##############

    # MVG MODELS
    evaluate_MVG_models(
        DTR_kfold_raw,
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_raw,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL
    )

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
        LVAL
    )

    ##############
    # SVM MODELS #
    ##############

    # Hyperparameter selection: Linear SVM
    SVM_C_selection(
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL
    )
    
    # Hyperparameter selection: Polynomial kernel SVM
    SVM_C_kernel_selection(
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL,
        KernelType.POLYNOMIAL
    )
    
    # Hyperparameter selection: RBF kernel SVM
    SVM_C_kernel_selection(
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL,
        KernelType.RBF
    )

    # Model evaluation
    evaluate_SVM_models(
        DTR_kfold_raw,
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_raw,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL
    )

    ##############
    # GMM MODELS #
    ##############
    
    # Hyperparameter selection
    GMM_components_selection(
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL
    )

    #Model evaluation
    evaluate_GMM_models(
        DTR_kfold_raw,
        DTR_kfold_gaussianized,
        DTR_kfold_z_normalized,
        LTR_kfold,
        DVAL_kfold_raw,
        DVAL_kfold_gaussianized,
        DVAL_kfold_z_normalized,
        LVAL
    )

    
    ####################
    # Model Comparison #
    ####################
    
    best_models = train_best_models(DTR_kfold_z_normalized, LTR_kfold, DVAL_kfold_z_normalized, LVAL)
    
    act_vs_min_DCF(best_models, LVAL)
    
    # Score recalibration
    
    best_models_recalibrated = recalibrate_models(best_models, LVAL)
    
    act_vs_min_DCF(best_models_recalibrated, LVAL)