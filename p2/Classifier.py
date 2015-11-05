import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

class Classifier:
    means = None
    stds = None
    training_set_row_count = 0

    features_to_remove = []

    def __init__(self, filename_training_data):
        np.random.seed(42)

        # Read in training data and separate appropriately
        self.train_raw = np.genfromtxt(filename_training_data, delimiter=',')
        np.random.shuffle(self.train_raw)
        self.training_set_row_count = self.train_raw.shape[0]
        self.train_ids = self.train_raw[:, 0:1]
        self.train_features = self.train_raw[:, 1:-1]
        self.train_labels = self.train_raw[:, -1:self.train_raw.shape[1]]

    def write_output_file(self, ids, predictions, filename):
        """Write given id-s and predictions to filename with headers."""
        assert(ids.shape[0] == predictions.shape[0])
        f = open(filename, 'w')
        f.write("Id,Delay\n")
        for i in range(0, predictions.shape[0]):
            f.write("%d,%d\n" % (ids[i], predictions[i]))
        f.close()

    def cross_validate(self):
        # TODO http://scikit-learn.org/stable/modules/cross_validation.html
        return

    @staticmethod
    def tSNE_plot2(features, labels):
        """Do dimension reduction to 2 dimensions with t-SNE and plot the result."""
        model = TSNE(n_components=2, random_state=0)
        transformed = model.fit_transform(features)

        print("Data transformed, now plotting...")

        # Plotting
        x = transformed[:, 0]
        y = transformed[:, 1]
        colors = labels

        plt.scatter(x, y, c=colors)
        plt.show()

    @staticmethod
    def tSNE_plot3(features, labels):
        """Do dimension reduction to 2 dimensions with t-SNE and plot the result."""
        model = TSNE(n_components=3, random_state=0)
        transformed = model.fit_transform(features)

        print("Data transformed, now plotting...")

        # Plotting
        x = transformed[:, 0]
        y = transformed[:, 1]
        z = transformed[:, 2]
        colors = labels

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, s=20, c=colors, depthshade=True)
        plt.show()

    def test(self):
        """Test stuff"""
        self.tSNE_plot3(self.train_features, self.train_labels)

    def run(self):
        """Train model and predict on test set."""

Classifier('data/train.csv').test()