import numpy as np
import sklearn
from sklearn.manifold import TSNE

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
        self.train_labels.shape = self.train_labels.shape[0]

    def write_output_file(self, ids, predictions, filename):
        """Write given id-s and predictions to filename with headers."""
        assert(ids.shape[0] == predictions.shape[0])
        f = open(filename, 'w')
        f.write("Id,Label\n")
        for i in range(0, predictions.shape[0]):
            f.write("%d,%d\n" % (ids[i], predictions[i]))
        f.close()

    def cross_validate(self, C=1, cv_count=10):
        """Fit given model in 10-fold cross-validation and return accuracy scores."""
        # tsne = TSNE(n_components=2, random_state=0)
        model = sklearn.svm.LinearSVC(dual=False, C=C) # sklearn.svm.SVC(kernel="linear")
        data = self.train_features #tsne.fit_transform(self.train_features)
        scores = sklearn.cross_validation.cross_val_score(model, data, self.train_labels, cv=cv_count)

        return np.mean(scores), np.std(scores)

    def test_Cs(self):
        Cs = [0.1]#, 0.3, 1, 3, 10]

        for C in Cs:
            r_mean, r_std = self.cross_validate(C=C, cv_count=10)
            print("C=%.1f:\tmean=%.3f, std=%.3f" % (C, r_mean, r_std))

    def predict_on_testset(self, model, file_in, file_out):
        """Given a testset file, generate the corresponding predictions file."""
        # Read data in
        raw = np.genfromtxt(file_in, delimiter=',')
        ids = raw[:, 0:1]
        features = raw[:, 1:]

        # Predict
        predictions = model.predict(features)

        # Write output
        self.write_output_file(ids, predictions, file_out)

    @staticmethod
    def accuracy(true_classes, predicted_classes):
        """Evaluate the multi-class accuracy given predictions and ground truth."""
        return sklearn.metrics.accuracy_score(true_classes, predicted_classes)

    def test(self):
        """Test stuff"""
        # model = self.fit_SVM(self.train_features, self.train_labels)
        self.test_Cs()

    def run(self):
        """Train model and predict on test set."""
        # Train model
        model = sklearn.svm.LinearSVC(dual=False, C=1)
        model.fit(self.train_features, self.train_labels)

        self.predict_on_testset(model, "data/validate_and_test.csv", "predictions/submission.csv")


Classifier('data/train.csv').run()