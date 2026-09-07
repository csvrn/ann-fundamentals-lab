import numpy as np

def mse(y_act,y_pred):
    if y_pred.shape == y_act.shape:
        return np.mean(np.square(y_act - y_pred))
    else:
        raise ValueError("shapes of actual and predicted values must match")

def mse_derivative(y_act,y_pred):
    return 1/y_pred.size * (2 * (y_pred - y_act))