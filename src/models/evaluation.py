from models.train import train_mvg_models, train_log_reg_model, train_svm_model, train_gmm_models
from data_processing.dimensionality_reduction import pca
from visualization.print import print_separator
from data_processing.utils import project_data
from constants import APPLICATIONS, LABELS
from models.svm import KernelType
from models.mvg import MVGModel
import numpy


####################
# Model Evaluation #
####################


def evaluate_mvg(data: dict, LTR: numpy.ndarray, LTE: numpy.ndarray):
    
    print_separator("MVG EVALUATION")

    for DTE_name, (DTR, DTE) in data.items():
    
        for application in APPLICATIONS:
            
            models = train_mvg_models(
                DTR,
                LTR,
                DTE,
                LTE,
                LABELS,
                application,
            )

            print("# {} - {} #".format(DTE_name, application))
            for model_type, model_values in models.items():
                print("{}: {}".format(model_type, model_values["minDCF"]))

            if DTE_name == "Raw features":

                for components in [7, 6]:

                    # Apply PCA projection
                    pca_projection_matrix = pca(DTR, components)
                    DTR_pca = project_data(DTR, pca_projection_matrix)
                    DTE_pca = project_data(DTE, pca_projection_matrix)

                    # Evaluate Gaussian models
                    models = train_mvg_models(
                        DTR_pca,
                        LTR,
                        DTE_pca,
                        LTE,
                        LABELS,
                        application,
                    )

                    print("# {} - PCA: components = {} - {} #".format(DTE_name, components, application))
                    for model_type, model_values in models.items():
                        print("{}: {}".format(model_type, model_values["minDCF"]))


def evaluate_lr(data: dict, LTR: numpy.ndarray, LTE: numpy.ndarray, quadratic: bool):
    
    print_separator("LOGISTIC REGRESSION EVALUATION")
    
    for DTE_name, (DTR, DTE) in data.items():
    
        for application in APPLICATIONS:
            
            print("# {} - {} #".format(DTE_name, application))
            
            for πT, _, _ in APPLICATIONS:
                
                model = train_log_reg_model(
                    DTR,
                    LTR,
                    DTE,
                    LTE,
                    λ=1e-6,
                    πT=πT,
                    quadratic=quadratic,
                    application=application,
                )

                print("πT = {}: {:.3f}".format(πT, model["minDCF"]))
                
                
                
def evaluate_svm(data: dict, LTR: numpy.ndarray, LTE: numpy.ndarray, kernel_type: KernelType, kernel_params: tuple):

    print_separator("SVM EVALUATION")
    
    hyperparameters = {
        KernelType.NO_KERNEL: 1e-1,
        KernelType.POLYNOMIAL: 1.0,
        KernelType.RBF: 1e2
    }
    
    for DTE_name, (DTR, DTE) in data.items():
    
        for application in APPLICATIONS:
            
            print("# {} - {} #".format(DTE_name, application))
            
            for πT, _, _ in APPLICATIONS:
                
                model = train_svm_model(
                    DTR,
                    LTR,
                    DTE,
                    LTE,
                    1.0,
                    hyperparameters[kernel_type],
                    kernel_type,
                    kernel_params,
                    πT,
                    application=application
                )

                print("πT = {}: {:.3f}".format(πT, model["minDCF"]))

                
def evaluate_gmm(data: dict, LTR: numpy.ndarray, LTE: numpy.ndarray):

    print_separator("GMM EVALUATION")

    steps = {
        MVGModel.MVG: 3,
        MVGModel.TIED: 5,
        MVGModel.NAIVE: 5,
        MVGModel.TIED_NAIVE: 5,
    }

    for DTE_name, (DTR, DTE) in data.items():
    
        for application in APPLICATIONS:
            
            models = train_gmm_models(
                DTR,
                LTR,
                DTE,
                LTE,
                steps
            )

            print("# {} - {} #".format(DTE_name, application))
            for model_type, model_values in models.items():
                print("{}: {}".format(model_type, model_values["minDCF"]))