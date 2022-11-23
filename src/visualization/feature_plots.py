from data_processing.data_load import load_training, load_features
from data_processing.analytics import pearson_correlation, z_normalization
from data_processing.dimensionality_reduction import pca
import matplotlib.pyplot as plt
import numpy


def plot_features_histograms(features: list[str], samples: numpy.ndarray, labels: numpy.ndarray) -> None:
    """
    Plot histogram for all features

    Args:
        features (list[str]): array of features names
        samples (numpy.ndarray): array of samples
        labels (numpy.ndarray): array of numerical labels

    Returns:
        None
    """

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


def plot_feature_histogram(feature: str, samples: numpy.ndarray, labels: numpy.ndarray) -> None:
    """
    Plot histogram for only one feature

    Args:
        features (str): features name
        samples (numpy.ndarray): array of samples
        labels (numpy.ndarray): array of numerical labels

    Returns:
        None
    """

    # Get samples by class
    non_pulsars = samples[labels == 0]
    pulsars = samples[labels == 1]

    # Plot histogram
    plt.hist(non_pulsars, bins=50, density=True, label="Non pulsars", alpha=0.4)
    plt.hist(pulsars, bins=50, density=True, label="Pulsars", alpha=0.4)
    plt.xlabel(feature)
    plt.legend()
    plt.show()


def plot_feature_pairs_sctterplots(features: list[str], samples: numpy.ndarray, labels: numpy.ndarray) -> None:
    """
    Plot scatterplot for each pair of features

    Args:
        features (list[str]): array of features names
        samples (numpy.ndarray): array of samples
        labels (numpy.ndarray): array of numerical labels

    Returns:
        None
    """

    # Get samples by class
    non_pulsars = samples[:, labels == 0]
    pulsars = samples[:, labels == 1]

    # Plot scatterplots for all pairs of different features
    for i in range(len(features)):
        for j in range(len(features)):
            if i != j:

                plt.plot(non_pulsars[i], non_pulsars[j], linestyle="", marker=".", markersize=10, label="Non pulsars")
                plt.plot(pulsars[i], pulsars[j], linestyle="", marker=".", markersize=10, label="Pulsars")
                plt.xlabel(features[i])
                plt.ylabel(features[j])

                plt.legend()
                plt.show()


def plot_correlation_heatmap(samples: numpy.ndarray) -> None:
    """
    Plot heatmap of pearson correlation between features

    Args:
        samples (numpy.ndarray): array of samples

    Returns:
        None
    """

    pearson_matrix = pearson_correlation(samples)
    plt.imshow(pearson_matrix, cmap="gist_yarg", vmin=0, vmax=1)
    # Add correlation value on each square of the heatmap
    for x in range(len(pearson_matrix)):
        for y in range(len(pearson_matrix[x])):
            cor_color = "w" if pearson_matrix[x][y] > 0.5 else "k"
            plt.text(x, y, "%.2f" % pearson_matrix[x][y], color=cor_color, ha="center", va="center")
    plt.show()


if __name__ == "__main__":

    # Load data from file
    samples, labels = load_training()
    features = load_features()

    z_normalized_samples = z_normalization(samples)
    pca_samples = pca(samples, 4)

    # Correlation heatmaps
    plot_correlation_heatmap(samples)
    plot_correlation_heatmap(pca_samples)
    plot_correlation_heatmap(z_normalized_samples)

    # Plot histogram for all features
    plot_features_histograms(features, samples, labels)

    # Scatterplots for all pairs of different features
    plot_feature_pairs_sctterplots(features, samples, labels)
