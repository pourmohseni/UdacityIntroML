import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)


def stepFunction(t):
    """
    simple step function
    """
    if t >= 0:
        return 1
    return 0


def prediction(X, W, b):
    """
    makes a prediction using on the step-function activation
    :param X: inputs
    :param W: weights
    :param b: bias
    :return: prediction
    """
    return stepFunction((np.matmul(X, W) + b)[0])


def perceptronStep(X, y, W, b, learn_rate=0.01):
    """
    implements the perceptron trick. The function should receive as inputs the data X, the labels y, the weights W (
    as an array), and the bias b, update the weights and bias W, b, according to the perceptron algorithm,
    and return W and b.
    """
    for p in range(len(X)):
        y_bar = prediction(X[p], W, b)
        my_y = y[p]
        factor = np.sign(y[p] - y_bar) * learn_rate
        W[0] += X[p][0] * factor
        W[1] += X[p][1] * factor
        b += factor
    return W, b


def trainPerceptronAlgorithm(X, y, learn_rate=0.01, num_epochs=25):
    """
    runs the perceptron algorithm repeatedly on the dataset, and returns a few of the boundary lines obtained in the
    iterations, for plotting purposes. :param X: features :param y: labels :param learn_rate: :param num_epochs:
    :return:
    """
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2, 1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0] / W[1], -b / W[1]))
    return boundary_lines, W, b


learn_rate = 0.01
data = np.loadtxt('data.csv', delimiter=',')
X = data[:, :-1]
y = data[:, -1]
boundary_lines, W, b = trainPerceptronAlgorithm(X, y)

df = pd.DataFrame(X)
df['y'] = y
df_ones = df.loc[df['y'] == 1]
df_zeros = df.loc[df['y'] == 0]

plt.scatter(df_ones.iloc[:, 0], df_ones.iloc[:, 1], color='blue')
plt.scatter(df_zeros.iloc[:, 0], df_zeros.iloc[:, 1], color='red')
x_vals = np.array([df.iloc[:, 0].min(), df.iloc[:, 0].max()])
for line in range(len(boundary_lines)):
    plt.plot(x_vals, boundary_lines[line][0] * x_vals + boundary_lines[line][0], color='black')
plt.show()
W = np.array(np.random.rand(2, 1))
b = np.random.rand(1)[0]
