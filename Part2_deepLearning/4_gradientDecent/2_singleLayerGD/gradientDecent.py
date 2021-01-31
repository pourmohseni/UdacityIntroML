import numpy as np
import pandas as pd
from data_prep import features, targets, features_test, targets_test


"""The following implement gradient descent and trains a network on the given data. Your goal here is to train the 
network until you reach a minimum in the mean square error (MSE) on the training set. """


def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))


def train_nn(features, targets, epochs, learnrate):
    # Use to same seed to make debugging easier
    np.random.seed(42)

    n_records, n_features = features.shape
    last_loss = None

    # Initialize weights
    weights = np.random.normal(scale=1 / n_features ** .5, size=n_features)

    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features.values, targets):
            # Loop through all records, x is the input, y is the target

            # Note: We haven't included the h variable from the previous
            #       lesson. You can add it if you want, or you can calculate
            #       the h together with the output

            # Calculate the output
            output = sigmoid(np.dot(x, weights))

            # Calculate the error
            error = y - output

            # Calculate the error term
            error_term = (y - output) * sigmoid_prime(x)

            # Calculate the change in weights for this sample mand add it to the total weight change
            del_w += error_term * x

        # Update weights using the learning rate and the average change in weights
        weights += learnrate * del_w / features.shape[0]

        # Printing out the mean square error on the training set
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, weights))
            loss = np.mean((out - targets) ** 2)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
    return weights



# Neural Network hyperparameters
epochs = 1000
learnrate = 0.5

# train the NN
weights = train_nn(features, targets, epochs, learnrate)

# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)

print("Prediction accuracy: {:.3f}".format(accuracy))