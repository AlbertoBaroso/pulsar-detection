import numpy

def kfold(DTR: numpy.ndarray, LTR: numpy.ndarray, k: int):
    """
    K-Fold cross validation: partition the dataset in k folds

    Args:
        DTR (numpy.ndarray):    Training data
        LTR (numpy.ndarray):    Labels of the training samples
        k (int):                Number of folds

    Returns:
        training_data (numpy.ndarray): k-1 folds of the training data
        training_labels (numpy.ndarray): labels corresponding to the training data
        validation_data (numpy.ndarray): 1 out of k folds as validation data
        validation_labels (numpy.ndarray): labels corresponding to the validation data
        
    """

    n_samples = DTR.shape[1]
    fold_sizes = numpy.array([n_samples // k] * k)
    fold_sizes[: n_samples % k] += 1
    last_end = 0

    for i in range(k):
        begin, end = last_end, last_end + fold_sizes[i]
        training_data = numpy.concatenate((DTR[:, :begin], DTR[:, end:]), axis=1)
        training_labels = numpy.concatenate((LTR[:begin], LTR[end:]))
        validation_data = DTR[:, begin:end]
        validation_labels = LTR[begin:end]
        last_end = end
        yield training_data, training_labels, validation_data, validation_labels