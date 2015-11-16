import numpy as np
import sklearn
from Visualiser import Visualiser
from sklearn.manifold import TSNE
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

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

        # Normalise
        #self.train_features = self.normalise(self.train_features)
        #self.train_features = sklearn.preprocessing.normalize(self.train_features, axis=0)

        # Add nonlinear features
        self.train_features = self.add_nonlinear_features(self.train_features)

        #Visualiser.tSNE_plot2(self.train_features, self.train_labels)

    def normalise(self, X):
        """Return normalised feature matrix."""
        norm_col = np.mean(X, axis=1)
        norm_col.shape = (X.shape[0], 1)
        norm_mat = np.repeat(norm_col, X.shape[1], axis=1)
        return X / norm_mat

    def write_output_file(self, ids, predictions, filename):
        """Write given id-s and predictions to filename with headers."""
        assert(ids.shape[0] == predictions.shape[0])
        f = open(filename, 'w')
        f.write("Id,Label\n")
        for i in range(0, predictions.shape[0]):
            f.write("%d,%d\n" % (ids[i], predictions[i]))
        f.close()

    def cross_validate(self, C=1, cv_count=10, gamma=1):
        """Fit given model in 10-fold cross-validation and return accuracy scores."""
        model = sklearn.svm.SVC(kernel="sigmoid", C=C)
        data = self.train_features
        scores = sklearn.cross_validation.cross_val_score(model, data, self.train_labels, cv=cv_count)

        return np.mean(scores), np.std(scores)

    def grid(self):
        """Do a grid search on a bunch of parameters."""
        parameters = {
                      'C': [1, 10, 100, 1000, 1e4],
                      'gamma': [1e-5, 1e-4, 0.001, 0.01, 0.1],
                      #'degree': [2, 3]
        }
        svr = sklearn.svm.SVC(kernel='rbf')
        clf = sklearn.grid_search.GridSearchCV(svr, parameters, cv=3, verbose=1, n_jobs=4)
        clf.fit(self.train_features, self.train_labels)

        for score in clf.grid_scores_:
            print(score)
        print("Best score: " + "\033[0;31m" + str(clf.best_score_) + "\033[0m")
        print(clf.best_params_)

    def grid_bag(self):
        """Train a bag model."""
        parameters = {
                      # 'max_samples': [0.4, 0.6, 0.8],
                      # 'max_features': [0.4, 0.6, 0.8, 1]
                        'n_estimators': [20, 30, 40, 50, 60, 100],
                        'max_features': [1, 2, 3, 4, "sqrt", "log2"],
                        'min_samples_split': [1, 3, 5],
        }

        bagging = RandomForestClassifier(max_depth=None, random_state=0)
        # bagging = sklearn.naive_bayes.GaussianNB()
        clf = sklearn.grid_search.GridSearchCV(bagging, parameters, cv=3, verbose=1, n_jobs=4)
        clf.fit(self.train_features, self.train_labels)

        for score in clf.grid_scores_:
            print(score)
        print("Best score: " + "\033[0;31m" + str(clf.best_score_) + "\033[0m")
        print(clf.best_params_)


    def predict_on_testset(self, model, file_in, file_out):
        """Given a testset file, generate the corresponding predictions file."""
        # Read data in
        raw = np.genfromtxt(file_in, delimiter=',')
        ids = raw[:, 0:1]
        features = raw[:, 1:]

        # Normalise features and add nonlinear ones
        features = self.normalise(features)
        features = self.add_nonlinear_features(features)

        # Predict
        predictions = model.predict(features)

        # Write output
        self.write_output_file(ids, predictions, file_out)

    @staticmethod
    def accuracy(true_classes, predicted_classes):
        """Evaluate the multi-class accuracy given predictions and ground truth."""
        return sklearn.metrics.accuracy_score(true_classes, predicted_classes)

    @staticmethod
    def add_nonlinear_features(features):
        """Calculate some additional features and return the original features concatenated with the new ones."""
        logarithms = np.log(features + 1)
        xlogx = np.multiply(features, logarithms)
        sqrts = np.sqrt(features)
        poly3 = np.power(features, 3)
        poly4 = np.power(features, 4)

        num_original_features = features.shape[1]
        polynomials = np.zeros(shape=(features.shape[0], num_original_features ** 2))
        for i in range(0, num_original_features):
            for j in range(0, num_original_features):
                feature1 = features[:, i]
                feature2 = features[:, j]
                polynomials[:, i * num_original_features + j] = np.multiply(feature1, feature2)

        return np.concatenate((features, logarithms, sqrts, xlogx, poly3, poly4, polynomials), axis=1)

    def test(self):
        """Test stuff"""
        # model = self.fit_SVM(self.train_features, self.train_labels)
        self.test_Cs()

    def run(self):
        """Train model and predict on test set."""
        # Train model
        #model = sklearn.svm.SVC(kernel="rbf", C=10, gamma=0.1)
        model = RandomForestClassifier(max_depth=None, max_features=2, min_samples_split=5, n_estimators=40)
        model.fit(self.train_features, self.train_labels)

        self.predict_on_testset(model, "data/validate_and_test.csv", "predictions/submission.csv")

    def plot_failures(self):
        """Fit a model and plot failing data points."""
        X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
            self.train_features, self.train_labels, test_size=0.6, random_state=0)

        gamma_modifier = 1
        model = sklearn.svm.SVC(kernel="rbf", C=1000)
        model.fit(X_train, y_train)

        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)
        failures = test_predictions != y_test

        print("Accuracy\tTEST: %.3f\tTRAIN:%.3f" %
              (self.accuracy(y_test, test_predictions), self.accuracy(y_train, train_predictions)))
        Visualiser.plot_failures(X_test, y_test, failures)

Classifier('data/train.csv').run()
# Classifier('data/train.csv').plot_failures()
# Classifier('data/train.csv').grid()
# Classifier('data/train.csv').grid_bag()
