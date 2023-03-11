import numpy

def print_separator(title: str = None):
    """
    Print a separation line to stdout

    Args:
        title (str, optional): Optional title to display in the middle of the separator. Defaults to None.
    """
    
    SEPARATOR_LENGTH = 60
    SEPARATOR = "#"
    print("\n" + SEPARATOR * SEPARATOR_LENGTH)
    
    if title is not None:
    
        additional_separators = (SEPARATOR_LENGTH - len(title) - 2)
        semi_separator_len = additional_separators // 2
        extra_separator = additional_separators % 2
        
        print(SEPARATOR * semi_separator_len + " " + title + " " + SEPARATOR * (semi_separator_len + extra_separator))
        print(SEPARATOR * SEPARATOR_LENGTH + "\n")
        
        
  
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