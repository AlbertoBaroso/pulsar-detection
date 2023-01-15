from data_processing.dimensionality_reduction import pca
from visualization.print import print_separator
from data_processing.utils import project_data
from constants import APPLICATIONS, K, LABELS
from data_processing.validation import kfold
from models.train import mvg_kfold
import numpy


def PCA_m_selection(training_samples: numpy.ndarray, training_labels: numpy.ndarray):
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