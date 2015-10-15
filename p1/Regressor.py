import numpy as np
import math


class Regressor:
    means = None
    stds = None
    training_set_row_count = 0

    def __init__(self, filename_training_data):
        np.random.seed(42)

        # Read in training data and separate appropriately
        self.train_raw = np.genfromtxt(filename_training_data, delimiter=',')
        np.random.shuffle(self.train_raw)
        self.training_set_row_count = self.train_raw.shape[0]
        self.train_ids = self.train_raw[:, 0:1]
        self.train_features = self.train_raw[:, 1:-1]
        self.train_labels = self.train_raw[:, -1:self.train_raw.shape[1]]

        # Normalise features
        self.means = np.mean(self.train_features, axis=0)
        self.stds = np.std(self.train_features, axis=0)
        train_features = self.normalised(self.train_features)

        # Add bias term
        bias = np.ones(shape=(self.train_raw.shape[0], 1))
        train_features = np.concatenate((bias, train_features), axis=1)

    def write_output_file(self, ids, predictions, filename):
        """Write given id-s and predictions to filename with headers."""
        assert(ids.shape[1] == predictions.shape[1])
        array = np.concatenate((ids, predictions), axis=1)
        np.savetxt(filename, array, header= "Id,Delay", delimiter=",", fmt=["%d", "%d"])

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

    def get_model(self, X, y, lamb=0):
        """Fit a ridge regression model and return parameters."""
        n_col = X.shape[1]
        return np.linalg.lstsq(X.T.dot(X) + lamb * np.identity(n_col), X.T.dot(y))

    def get_model2(self, X, y, lamb=0):
        n_col = X.shape[1]
        return np.linalg.inv(X.T.dot(X) + lamb * np.identity(n_col)).dot(X.T).dot(y).T

    def predict(self, params, X):
        """Predict y given X and parameters."""
        return X.dot(params)

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

    def fit(self, subslice, lamb=0):
        """Fit one linear regression model using rows given in row_inds and return the model and loss."""

        # Get data
        row_start, row_end = subslice
        train_features, train_labels, test_features, test_labels = self.cv_separate_data(subslice)

        # Fit model
        model = self.get_model(train_features,
                               train_labels,
                               lamb=lamb)
        params = model[0]
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
        print("RMSE: " + str(int(rmse)) + "\t\tLoss: " + str(int(loss)))

        return model, rmse, loss

    def cross_validate(self, cv_count):
        """Train cv_count models and test by partitioning the dataset into cv_count slices."""

        best_loss = float('inf')
        best_rmse = None
        best_model = None
        best_slice = None

        lambs = [0, 1, 100, 1000, 10000, 100000, 1000000]

        for i in range(0, cv_count):
            lamb = lambs[i]

            # Slice training set
            slice_size = self.training_set_row_count / cv_count
            slice_start = i * slice_size
            slice_end = (i + 1) * slice_size
            if i == cv_count - 1:
                slice_end = self.training_set_row_count

            # Fit model
            print("Cross-validation: slice %d of %d (%d rows in test)." % (i, cv_count, slice_end-slice_start))
            model, rmse, loss = self.fit((slice_start, slice_end), lamb=lamb)

            # Check if current model is better than previous best
            if(loss < best_loss):
                best_loss = loss
                best_rmse = rmse
                best_model = model
                best_slice = i

        print("CV result: slice %d with RMSE %d and lambda %d." % (best_slice, best_rmse, lambs[best_slice]))

    def predict_on_testset(self, model, file_in, file_out):
        """Given a testset file, generate the corresponding predictions file."""



    def test(self):
        """Test stuff"""
        # self.fit((1, 600))
        self.cross_validate(5)


Regressor('data/train.csv').test()