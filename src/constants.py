# General

K = 10  # Number of folds for cross validation
LABELS = [0, 1]
APPLICATIONS = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)] 
 
# Preprocessing

PCA_COMPONENTS = 4

# Logistic Regression

LOG_REG_λ = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]

# Support Vector Machines

SVM_C = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
SVM_c = [1e-1, 1e-0, 1e1]
SVM_γ = [1e-3, 1e-2, 1e-1]
SVM_K = 1

# Gaussian Mixture Models

GMM_STEPS = [1, 2, 3, 4, 5, 6]