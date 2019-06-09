# q1
import sklearn
from sklearn.datasets import load_boston
import numpy as np


def compute_huber_loss(a, delta):
    huber_loss = np.copy(a)
    # y1: y indexes for |a| <= delta
    # y2: y indexes for |a| > delta
    y1 = np.where(np.absolute(a) <= delta)
    y2 = np.where(np.absolute(a) > delta)

    huber_loss[y1] = 0.5 * np.power((a)[y1], 2)
    huber_loss[y2] = delta * (np.absolute((a)[y2]) - 0.5 * delta)

    return huber_loss


# full batch mdde mode gradient decsend
def huber_gradient_descent(x, t, epochs, learning_rate, delta):
    w = np.zeros(len(x[0]))
    b = 0
    N = x.shape[0]
    cost_history = np.zeros(epochs)

    for iteration in range(0, epochs):
        # current estimates
        y = np.dot(x, w) + b

        # compute gradients
        huber_loss_gradient = compute_huber_gradient(y - t, delta)
        dJ_dw = (1 / N) * np.dot(np.transpose(x), huber_loss_gradient)
        dJ_db = (1 / N) * np.sum(huber_loss_gradient)

        # update parameters
        w -= learning_rate * dJ_dw
        b -= learning_rate * dJ_db

        # record cost function
        L = compute_huber_loss(y - t, delta)
        J = (1 / N) * np.sum(L)
        cost_history[iteration] = J


def compute_huber_gradient(a, delta):
    huber_loss_gradient = np.copy(a)

    # y1: y indexes for |a| <= delta
    # y2: y indexes for |a| > delta
    y1 = np.where(np.absolute(a) <= delta)
    y2 = np.where(np.absolute(a) > delta)

    huber_loss_gradient[y1] = a[y1]
    huber_loss_gradient[y2] = delta * a[y2] / np.absolute(a[y2])

    return huber_loss_gradient


if __name__ == "__main__":
    # w = 0
    # load dataset
    boston = load_boston()
    x = boston.data
    t = boston.target

    # learning parameters
    epochs = 60
    learning_rate = 0.00001
    delta = 2
    huber_gradient_descent(x, t, epochs, learning_rate, delta)
