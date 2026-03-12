import matplotlib.pyplot as plt
import numpy as np

def plot_tree_feature_importances(feature_names, model):
    """
    Plot feature importances from tree model
    """
    n_features = model.feature_importances_.size
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)