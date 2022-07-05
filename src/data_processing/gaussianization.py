from scipy.stats import norm
import numpy

### GAUSSIANIZATION ###

# Compute percentage of samples smaller than x
def rank_samples(feature, training_samples):
    comparison_feature = feature if training_samples is None else training_samples
    ranks = []
    for x in feature:
        less_than_x = 0
        for x_i in comparison_feature: # x_i is the value of the considered feature for the i-th training sample
            if x_i < x:
                less_than_x += 1
        rank = (less_than_x + 1) / (feature.shape[0] + 2)
        ranks.append(rank)
    # +2 because we assume the existance of a feature smaller than all the others and a feature larger than all the others)
    return ranks

# Compute percentile (percent poin function) of the rank
def rank_percentile(rank):
    return norm.ppf(rank)

def gaussianize_feature(feature, training_samples=None):
    return numpy.array(rank_percentile(rank_samples(feature, training_samples)))
    
def gaussianize_training_samples(DTR):
    return numpy.array([gaussianize_feature(x) for x in DTR])

def gaussianize_test_features(DTE, DTR):
    # Compute gaussianization by raking test samples against training samples
    result = numpy.array([])
    for i in range(len(DTE)):
        result = numpy.append(result, gaussianize_feature(DTE[i], DTR[i]))
    return result
    
if __name__ == '__main__': 
    
    import sys
    sys.path.append("./")
    from data_processing.data_load import load_training, load_features, load_test
    from visualization.feature_plots import plot_feature_pairs_sctterplots
    import matplotlib.pyplot as plt
    
    features = load_features()
    training_samples, training_labels = load_training()
    for i in range(len(features)):
    
        gaussianized_training_samples = gaussianize_feature(training_samples[i])
        non_pulsars = gaussianized_training_samples[training_labels == 0]
        pulsars = gaussianized_training_samples[training_labels == 1]
    
        plt.hist(non_pulsars, bins=50, density=True, label="Non pulsars", alpha=0.4)
        plt.hist(pulsars, bins=50, density=True, label="Pulsars", alpha=0.4)
        plt.xlabel(features[i])
        plt.legend()
        plt.show()