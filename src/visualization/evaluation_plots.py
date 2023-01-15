from data_processing.comparison import optimal_bayes_decisions, normalized_DCF, minimum_DCF
from data_processing.analytics import confusion_matrix
import matplotlib.pyplot as plt
from constants import LOG_REG_λ
import numpy


def bayes_error_plot(scores: numpy.ndarray, labels: numpy.ndarray) -> None:
    """
    Draw Bayes error plot

    Args:
        scores (numpy.ndarray): Scores assigned to the test samples
        labels (numpy.ndarray): Labels of the test samples
    """

    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    DCF, min_DCF = [], []

    for p in effPriorLogOdds:
        π_tilde = 1.0 / (1.0 + numpy.exp(-p))
        predictions = optimal_bayes_decisions(scores, π_tilde, 1, 1)
        CM = confusion_matrix(predictions, labels)
        DCF.append(normalized_DCF(CM, π_tilde, 1, 1))
        min_DCF.append(minimum_DCF(scores, labels, π_tilde, 1, 1))

    plt.plot(effPriorLogOdds, DCF, label="DCF", color="r")
    plt.plot(effPriorLogOdds, min_DCF, label="min DCF", color="b")
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF value")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()


def plot_minDCF_LR(title: str, applications: dict):

    colors = ["r", "b", "g"]

    for i, (application, minDCFs) in enumerate(applications.items()):

        plt.plot(LOG_REG_λ, minDCFs, label=r'minDCF($\tilde{\pi}$ = ' + str(application[0]) + ")", color=colors[i])

    plt.title(title)
    plt.xlabel("λ")
    plt.ylabel("minDCF")
    plt.xscale('log')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()