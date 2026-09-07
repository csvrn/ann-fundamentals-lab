import numpy as np

def relu(z):
    return z

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_derivative(z):
    return np.exp(-z) / (1 + np.exp(-z))**2

def linear(z):
    return z

def linear_derivative(z):
    return np.ones_like(z)