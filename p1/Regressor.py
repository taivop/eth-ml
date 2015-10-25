import numpy as np
import math
from sklearn import linear_model


class Regressor:
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

        # Remove some features
        self.train_features = self.delete_original_features(self.train_features)

        # Add nonlinear features
        self.train_features = self.add_nonlinear_features(self.train_features)

        # Normalise features
        self.means = np.mean(self.train_features, axis=0)
        self.stds = np.std(self.train_features, axis=0)
        self.train_features = self.normalised(self.train_features)

        # Add bias term
        bias = np.ones(shape=(self.train_raw.shape[0], 1))
        self.train_features = np.concatenate((bias, self.train_features), axis=1)

        # Find and remove redundant features
        self.find_redundant_features()
        self.train_features = self.remove_redundant_features(self.train_features)

    def delete_original_features(self, features):
        """Remove a subset of the original features"""
        return np.delete(features, [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 14], 1)

    def find_redundant_features(self, lamb=0.1, vocal=False):
        """Remove a subset of all features"""

        cv_count = 10

        original_features = np.copy(self.train_features)

        # Calculate baseline RMSE using all features
        baseline = self.cross_validate(cv_count, lamb=lamb, vocal=False)[1]

        to_remove = []

        # Remove features one by one and compare RMSE with baseline
        for i in range(0, self.train_features.shape[1]):
            self.train_features = np.delete(self.train_features, i, 1)
            rmse = self.cross_validate(cv_count, lamb=lamb, vocal=False)[1]

            if rmse - baseline <= -0.05:
                to_remove.append(i)

            # Restore original features
            self.train_features = np.copy(original_features)

            if vocal:
                print("Removing feature %3d: change in RMSE is %5.1f" % (i, rmse - baseline))

        # Remove whole feature set that we found and compare with baseline
        self.train_features = np.delete(self.train_features, to_remove, 1)
        rmse = self.cross_validate(cv_count, lamb=lamb, vocal=False)[1]
        self.train_features = np.copy(original_features)

        if vocal:
            print("Removing feature set " + str(to_remove) + ":")
            print("RMSE\t old: %5.1f new: %5.1f" % (baseline, rmse))

        self.features_to_remove = to_remove

    def remove_redundant_features(self, features):
        """Remove redundant features we have found"""
        return np.delete(features, self.features_to_remove, 1)

    def add_nonlinear_features(self, features):
        """Calculate some additional features and return the original features concatenated with the new ones."""
        logarithms = np.log(features + 1)
        #powers = np.power(np.ones(shape=features.shape) * 2, features)
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

        return(np.concatenate((features, logarithms, sqrts, xlogx, poly3, poly4, polynomials), axis=1))
        # return features

    def write_output_file(self, ids, predictions, filename):
        """Write given id-s and predictions to filename with headers."""
        assert(ids.shape[0] == predictions.shape[0])
        f = open(filename, 'w')
        f.write("Id,Delay\n")
        for i in range(0, predictions.shape[0]):
            f.write("%d,%d\n" % (ids[i], predictions[i]))
        f.close()

    def write_parameters(self, params, filename):
        """Write parameter array to file"""
        np.savetxt(filename, params)

    def read_parameters(self, filename):
        """Read parameter array from file"""
        return np.genfromtxt(filename).reshape((-1,1))

    def normalised(self, data):
        """For each column in array, move the means to 0 and divide by standard deviation."""
        means_tiled = np.tile(self.means, (data.shape[0], 1))
        stds_tiled = np.tile(self.stds, (data.shape[0], 1))
        return ((data - means_tiled) / stds_tiled)

    def denormalised(self, data):
        """For each column in array, apply reverse normalisation."""
        means_tiled = np.tile(self.means, (data.shape[0], 1))
        stds_tiled = np.tile(self.stds, (data.shape[0], 1))
        return (data * stds_tiled + means_tiled)

    def get_params(self, X, y, lamb=0):
        """Fit a ridge regression model and return parameters."""
        n_col = X.shape[1]
        return np.linalg.lstsq(X.T.dot(X) + lamb * np.identity(n_col), X.T.dot(y))[0]

    def get_params2(self, X, y, lamb=0):
        n_col = X.shape[1]
        model = np.linalg.inv(X.T.dot(X) + lamb * np.identity(n_col)).dot(X.T).dot(y).T
        arr = np.zeros(shape=(model[0].shape[0], 1))
        arr[:,0] = model[0]
        return np.reshape(model[0], newshape=(model[0].shape[0], 1))

    def get_params_lasso(self, X, y, lamb=0, max_iter=1000):
        """Fit a LASSO regression model and return parameters."""
        n_col = X.shape[1]
        clf = linear_model.Lasso(alpha=lamb, max_iter=max_iter, fit_intercept=False)
        clf.fit(X, y)
        arr = np.zeros(shape=(X.shape[1], 1))
        arr[:, 0] = clf.coef_
        return arr

    def predict(self, params, X):
        """Predict y given X and parameters."""
        return X.dot(params)

    def fit(self, lamb=0):
        """Fit one linear regression model using all rows and return the model and loss."""

        # Fit model
        params = self.get_params(self.train_features,
                               self.train_labels,
                               lamb=lamb)
        predictions = self.predict(params, self.train_features)

        # Calculate loss
        errors = predictions - self.train_labels
        loss = (errors.T.dot(errors) + lamb * params.T.dot(params))[0, 0] / errors.shape[0]
        rmse = math.sqrt(errors.T.dot(errors) / errors.shape[0])

        print("Lambda: " + str(lamb))
        print("RMSE: " + str(int(rmse)) + "\t\tLoss: " + str(int(loss)))

        return params, rmse, loss

    def cv_fit(self, subslice, lamb=0):
        """Fit one linear regression model using rows given in row_inds and return the model and loss."""

        # Get data
        row_start, row_end = subslice
        train_features, train_labels, test_features, test_labels = self.cv_separate_data(subslice)

        # Fit model
        params = self.get_params(train_features, train_labels,
                                       lamb=lamb)
        predictions = self.predict(params, test_features)

        # Calculate loss
        errors = predictions - test_labels
        loss = (errors.T.dot(errors) + lamb * params.T.dot(params))[0, 0] / errors.shape[0]
        rmse = math.sqrt(errors.T.dot(errors) / errors.shape[0])

        # print("Training set:\n" + str(self.train_features))
        # print("Training set labels:\n" + str(self.train_labels))
        # print("Parameters:\n" + str(params))
        # print("Predictions:\n" + str(predictions))
        # print("Ground truth:\n" + str(self.train_labels))
        # print("Side-by-side:\n" + str(np.concatenate((predictions, self.train_labels), axis=1)))
        # print("Lambda: " + str(lamb))
        # print("RMSE: " + str(int(rmse)) + "\t\tLoss: " + str(int(loss)))

        return params, rmse, loss

    def cv_separate_data(self, test_inds):
        """Separate data into test and train set for cross-validation."""
        test_start, test_end = test_inds

        test_features = self.train_features[test_start:test_end, :]
        test_labels = self.train_labels[test_start:test_end, :]
        train_features = np.concatenate((self.train_features[0:test_start, :],
                                         self.train_features[test_end:self.train_features.shape[0], :]
                                         ), axis=0)
        train_labels = np.concatenate((self.train_labels[0:test_start, :],
                                       self.train_labels[test_end:self.train_labels.shape[0], :]
                                       ), axis=0)

        return train_features, train_labels, test_features, test_labels

    def cross_validate(self, cv_count, lamb, vocal=True):
        """Train cv_count models and test by partitioning the dataset into cv_count slices."""

        best_loss = float('inf')
        best_rmse = None
        best_model = None
        best_slice = None

        rmses = []

        for i in range(0, cv_count):

            # Slice training set
            slice_size = self.training_set_row_count / cv_count
            slice_start = i * slice_size
            slice_end = (i + 1) * slice_size
            if i == cv_count - 1:
                slice_end = self.training_set_row_count

            # Fit model
            #print("Cross-validation: slice %d of %d (lamb=%.3f)." % (i, cv_count, lamb))
            model, rmse, loss = self.cv_fit((slice_start, slice_end), lamb=lamb)
            rmses.append(rmse)

            # Check if current model is better than previous best
            if(loss < best_loss):
                best_loss = loss
                best_rmse = rmse
                best_model = model
                best_slice = i

        #print("CV result: slice %d with RMSE %d and lambda %d." % (best_slice, best_rmse, lamb))
        if vocal:
            print("CV result: RMSE mean %d, stdev %d." % (np.mean(rmses), np.std(rmses)))

        return best_model, np.mean(rmses)

    def predict_on_testset(self, params, file_in, file_out):
        """Given a testset file, generate the corresponding predictions file."""
        # Read data in
        raw = np.genfromtxt(file_in, delimiter=',')
        ids = raw[:, 0:1]
        features = raw[:, 1:]

        # Delete some features
        features = self.delete_original_features(features)

        # Add nonlinear features
        features = self.add_nonlinear_features(features)

        # Normalise data
        features = self.normalised(features)

        # Add bias term
        bias = np.ones(shape=(features.shape[0], 1))
        features = np.concatenate((bias, features), axis=1)

        # Remove redundant features
        features = self.remove_redundant_features(features)

        # Predict
        predictions = self.predict(params, features)

        # Write output
        self.write_output_file(ids, predictions, file_out)

    def test(self):
        """Test stuff"""
        #for lamb in [0, 0.001, 0.01, 0.03, 0.06, 0.1, 0.2, 0.5, 1, 3, 6, 10, 30, 100, 300, 1000]:
        for lamb in [0, 0.01, 0.1, 1, 3, 5, 7, 9]:
            print("---- LAMBDA = %.3f ----" % (lamb))
            params = self.cross_validate(10, lamb)

    def run(self):
        params = self.fit(lamb=0.1)[0]
        predictions = self.predict(params, self.train_features)

        # Calculate loss
        errors = predictions - self.train_labels
        rmse = math.sqrt(errors.T.dot(errors) / errors.shape[0])
        #print(rmse)
        self.predict_on_testset(params, 'data/validate_and_test.csv', 'predictions/validate_and_test.out')

Regressor('data/train.csv').test()
Regressor('data/train.csv').run()

# Regressor('data/train.csv').find_redundant_features()