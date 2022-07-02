import matplotlib.pyplot as plt
import sys
sys.path.append("./")
from utils.data_load import load_training, load_features

# Compute mean, min and max value of each feature
def compute_analytics(features, samples):
    print("Feature \t\t\t\t\t│  Mean \t│  Min \t\t│  Max")
    print("─" * 90)
    for i, feature in enumerate(features):
        feature_name = feature + (" " * (40 - len(feature)))
        mean_value = samples[i].mean()
        min_value = samples[i].min()
        max_value = samples[i].max()
        print("{} \t│ {:.2f}  \t│ {:.2f}  \t│ {:.2f}".format(feature_name, mean_value, min_value, max_value))


if __name__ == '__main__':

    # Load data from file
    samples, labels = load_training()
    features = load_features()
    
    compute_analytics(features, samples)