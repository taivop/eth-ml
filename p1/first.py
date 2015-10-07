import numpy as np



def write_output_file(array, filename):
	"""Write given 2-column array to filename with headers."""
	assert(array.shape[1] == 2)
	np.savetxt(filename, array, header= "Id,Delay", delimiter=",")


def normalised(data, means, stds):
	"""For each column in array, move the means to 0 and divide by standard deviation."""
	means_tiled = np.tile(means, (data.shape[0], 1))
	stds_tiled = np.tile(stds, (data.shape[0], 1))
	return ((data - means_tiled) / stds_tiled)

def denormalised(data, means, stds):
	"""For each column in array, apply reverse normalisation."""
	means_tiled = np.tile(means, (data.shape[0], 1))
	stds_tiled = np.tile(stds, (data.shape[0], 1))
	return (data * stds_tiled + means_tiled)

def get_model(X, y, lamb=0):
    n_col = X.shape[1]
    return np.linalg.lstsq(X.T.dot(X) + lamb * np.identity(n_col), X.T.dot(y))

def predict(params, X):
	return(X.dot(params))



def fit():
	"""Fit one linear regression model and return accuracy."""
	# Read data file
	train_raw = np.genfromtxt('data/train.csv', delimiter=',')
	train_ids = train_raw[:, 0:1]
	train_features = train_raw[:, 1:-1]
	train_labels = train_raw[:, -1:train_raw.shape[1]]

	# Normalise features
	means_train = np.mean(train_features, axis=0)
	stds_train = np.std(train_features, axis=0)
	train_features = normalised(train_features, means_train, stds_train)

	# TODO: fit model
	model = get_model(train_features, train_labels, lamb=0)
	params = model[0]
	predictions = np.round(predict(params, train_features))

	print(np.concatenate((predictions, train_labels), axis=1))

	# TODO: calculate accuracy


def test():
	"""Test stuff"""
	# Test writing 
	write_output_file(np.asarray([[1, 200],[2, 35]]), "data/output_test.csv")


fit()