import numpy as np


class Regressor:
    means = None
    stds = None

    def __init__(self, filename_training_data):

        # Read in training data and separate appropriately
        self.train_raw = np.genfromtxt(filename_training_data, delimiter=',')
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


    def fit(self):
        """Fit one linear regression model and return the model and loss."""

        lamb = 0

        # Fit model
        model = self.get_model(self.train_features, self.train_labels, lamb=lamb)
        params = model[0]
        predictions = self.predict(params, self.train_features)

        # Calculate loss
        errors = predictions - self.train_labels
        loss = (errors.T.dot(errors) + lamb * params.T.dot(params))[0, 0]

        print("Training set:\n" + str(self.train_features))
        print("Training set labels:\n" + str(self.train_labels))
        print("Parameters:\n" + str(params))
        print("Predictions:\n" + str(predictions))
        print("Ground truth:\n" + str(self.train_labels))
        print("Side-by-side:\n" + str(np.concatenate((predictions, self.train_labels), axis=1)))
        print("Loss:\n" + str(loss))

        return model, loss

    def predict_on_testset(self, model, file_in, file_out):
        """Given a testset file, generate the corresponding predictions file."""


    def test(self):
        """Test stuff"""
        self.fit()


Regressor('data/train.csv').test()