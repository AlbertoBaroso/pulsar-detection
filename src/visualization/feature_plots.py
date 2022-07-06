import matplotlib.pyplot as plt
import sys
sys.path.append("./")
from data_processing.data_load import load_training, load_features
from data_processing.analytics import pearson_correlation
from data_processing.dimensionality_reduction import pca

# Plot histogram for all features
def plot_features_histograms(features, samples, labels):

    # Get samples by class
    non_pulsars = samples[:, labels == 0]
    pulsars = samples[:, labels == 1]

    # Plot histogram
    for i in range(len(features)):
        plt.hist(non_pulsars[i], bins=20, density=True, label="Non pulsars", alpha=0.4)
        plt.hist(pulsars[i], bins=20, density=True, label="Pulsars", alpha=0.4)
        plt.xlabel(features[i])
        plt.legend()
        plt.show()

# Plot histogram for only one feature
def plot_feature_histogram(feature, samples, labels):

    # Get samples by class
    non_pulsars = samples[labels == 0]
    pulsars = samples[labels == 1]

    # Plot histogram
    plt.hist(non_pulsars, bins=50, density=True, label="Non pulsars", alpha=0.4)
    plt.hist(pulsars, bins=50, density=True, label="Pulsars", alpha=0.4)
    plt.xlabel(feature)
    plt.legend()
    plt.show()

def plot_feature_pairs_sctterplots(features, samples, labels):

    # Get samples by class
    non_pulsars = samples[:, labels == 0]
    pulsars = samples[:, labels == 1]

    # Plot scatterplots for all pairs of different features
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j:

                plt.plot(non_pulsars[i], non_pulsars[j], linestyle='', marker='.', markersize=10, label="Non pulsars")
                plt.plot(pulsars[i], pulsars[j], linestyle='', marker='.', markersize=10, label="Pulsars")
                plt.xlabel(features[i])
                plt.ylabel(features[j])
                
                plt.legend()
                plt.show()


def plot_correlation_heatmap(samples):
    pearson_matrix = pearson_correlation(samples)
    plt.imshow(pearson_matrix, cmap='gist_yarg', vmin=-1, vmax=1)
    plt.show()
    


if __name__ == '__main__':

    # Load data from file
    samples, labels = load_training()
    features = load_features()
    
    pca_samples = pca(samples, 4, labels)
    
    plot_correlation_heatmap(samples)
    plot_correlation_heatmap(pca_samples)
    # plot_features_histograms(features, samples, labels)
    # plot_feature_pairs_sctterplots(features, samples, labels)
