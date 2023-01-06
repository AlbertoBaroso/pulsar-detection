from data_processing.comparison import optimal_bayes_decisions, normalized_DCF, minimum_DCF
from data_processing.analytics import confusion_matrix
import matplotlib.pyplot as plt
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
    
    
def print_confusion_matrix(confusion_matrix: numpy.ndarray) -> None:
    """
    Given a confusion matrix, print it in a readable format

    Args:
        confusion_matrix (numpy.ndarray): Array of shape (K, K) where K is the number of classes
    """
    print("\t\t\t  Actual")
    size = len(confusion_matrix)
    print("\t\t  |\t{}".format("\t".join(str(e) for e in list(range(size)))))
    print("\t\t--" + "-" * 8 * size)
    for k in range(size):
        if int(size / 2) == k:
            print("Predicted\t", end="")
        else:
            print("\t\t", end="")
        print("{} |\t{}".format(k, "\t".join(str(e) for e in confusion_matrix[k])))