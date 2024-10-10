import numpy as np


def local_weighted_regression(x0, X, Y, tau):
    # add bias term
    x0 = np.r_[1, x0]
    X = np.c_[np.ones(len(X)), X]

    # fit model: normal equations with kernel
    xw = X.T * weights_calculate(x0, X, tau)
    theta = np.linalg.pinv(xw @ X) @ xw @ Y
    # "@" is used to
    # predict value
    return x0 @ theta


# function to perform weight calculation
def weights_calculate(x0, X, tau):
    return np.exp(np.sum((X - x0) ** 2, axis=1) / (-2 * (tau**2)))


# plot locally weighted regression for different bandwidth values
def compute_predictions(X, Y, tau):
    predictions = [local_weighted_regression(x0, X, Y, tau) for x0 in X]

    return predictions


def compute_wQISA_predictions(Y, k):

    # These are actually the control points of a weighted quasi-interpolant spline approximation:
    kernel = np.full((k,), 1 / k)
    radius = (k - 1) // 2
    Y_ext = np.pad(Y.tolist(), (radius, radius), mode="edge")
    predictions = np.convolve(Y_ext, kernel, "valid")

    return predictions
