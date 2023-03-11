from visualization.evaluation_plots import bayes_error_plot
from models.train import logistic_regression_kfold
from data_processing.validation import kfold
from models.lr import LogisticRegression
from constants import  K
import numpy

######################
# Score Recalibation #
######################


def recalibrate_scores_kfold(scores: numpy.ndarray, labels: numpy.ndarray, π_tilde: float):
    """Train LR models with K-fold to recalibrate scores"""

    DTR_kfold, LTR_kfold, DVAL_kfold = numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray), numpy.empty(K, dtype=numpy.ndarray)
    for i, (DTR, LTR, DVAL, _LVAL) in enumerate(kfold(scores, labels, K)):
        DTR_kfold[i], LTR_kfold[i], DVAL_kfold[i] = DTR, LTR, DVAL
    
    # Train a LR model to recalibrate scores
    recalibrated_scores = logistic_regression_kfold(
        DTR_kfold, LTR_kfold, DVAL_kfold, labels, λ=0, πT=0.5, quadratic=False, application=(π_tilde, 1, 1), return_scores=True
    )
    
    return recalibrated_scores - numpy.log(π_tilde / (1 - π_tilde))

def recalibrate_scores(scores: numpy.ndarray, labels: numpy.ndarray, π_tilde: float):
    """Train a LR model to recalibrate scores"""

    lr = LogisticRegression(scores, labels, λ=0, πT=0.5, quadratic=False)
    α, β = lr.w, lr.b
    recalibrated_scores = α * scores + β - numpy.log(π_tilde / (1 - π_tilde))
    
    return recalibrated_scores.ravel(), α, β


def recalibrate_models(models: dict, labels: numpy.ndarray, kfold: bool = False):
    """Recalibrate models scores"""

    recalibration_mode = recalibrate_scores_kfold if kfold else recalibrate_scores

    for model, scores in models.items():
        models[model] = recalibration_mode(numpy.array([scores]), labels, π_tilde=0.5)
        print("{} recalibrated".format(model))
        
    return models


def act_vs_min_DCF(models, LVAL):
    """Draw Bayes error plots"""

    for model, scores in models.items():

        bayes_error_plot(model, scores, LVAL)