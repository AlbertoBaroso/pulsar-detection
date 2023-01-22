from data_processing.comparison import optimal_bayes_decisions, normalized_DCF, minimum_DCF
from data_processing.analytics import confusion_matrix
import matplotlib.pyplot as plt
import numpy


def bayes_error_plot(title: str, scores: numpy.ndarray, labels: numpy.ndarray) -> None:
    """
    Draw Bayes error plot

    Args:
        title  (str):           Title of the plot
        scores (numpy.ndarray): Scores assigned to the test samples
        labels (numpy.ndarray): Labels of the test samples
    """

    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    act_DCF, min_DCF = [], []

    for p in effPriorLogOdds:
        π_tilde = 1.0 / (1.0 + numpy.exp(-p))
        predictions = optimal_bayes_decisions(scores, π_tilde, 1, 1)
        CM = confusion_matrix(predictions, labels)
        act_DCF.append(normalized_DCF(CM, π_tilde, 1, 1))
        min_DCF.append(minimum_DCF(scores, labels, π_tilde, 1, 1))
        
    plt.plot(effPriorLogOdds, act_DCF, label="act DCF", color="r")
    plt.plot(effPriorLogOdds, min_DCF, label="min DCF", color="b")
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF value")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.title(title)
    plt.legend()
    plt.show()


def plot_minDCF(title: str, hyperparameter: str, x_axis: list, applications: dict, log_scale: bool = False, max_y: float = 1.1, param_name='$\tilde{\pi}$', extra = '0.5) - ') -> None:

    colors = ["r", "b", "g"]
    xi = x_axis if log_scale else list(range(len(x_axis)))

    for i, (application, minDCFs) in enumerate(applications.items()):

        param_value = str(application[0]) + ')' if type(application) is tuple else extra + str(application)
        
        plt.plot(xi, minDCFs, label=r'minDCF(' + param_name + ' = ' + param_value, color=colors[i])

    plt.title(title)
    plt.xlabel(hyperparameter)
    plt.ylabel("minDCF")
    if log_scale:
        plt.xscale('log')
    else:
        plt.xticks(xi, x_axis)
    plt.ylim([0, max_y])
    plt.legend()
    plt.show()