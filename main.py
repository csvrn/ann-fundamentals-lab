from src.neuralnetwork import NeuralNetwork
from src.layer import Layer
from src.neuron import Neuron
from src.activation import *
from src.loss import mse
import numpy as np
import random
import math
def main():

    # XOR Example

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    weights_1 = np.array([
        [0.2, -0.4],
        [0.7,  0.3]
    ])

    bias_1 = np.array([
        0.1,
        -0.2
    ])

    weights_2 = np.array([
        [0.5, -0.6]
    ])

    bias_2 = np.array([
        0.2
    ])

    l1 = Layer(2, weights_1, 2, sigmoid, bias_1)
    l2 = Layer(2, weights_2, 1, sigmoid, bias_2)

    ann = NeuralNetwork([l1, l2])

    loss, output_pred = ann.run(
        X.T,
        y.T,
        0.5,
        sigmoid_derivative,
        sigmoid_derivative
    )
    print("XOR EXAMPLE:\n")
    print(f"loss: {loss}")
    print(f"output_pred:\n{output_pred}\n")

    #-------------------------------------------------------
    # Sine Example
    samples = [random.uniform(0, 360) for _ in range(10)]
    # samples = [163.97516125708387, 171.23715140693645, 257.86437428024084, 78.7200438545211, 64.02367312090713, 312.2374392355466, 24.9004858030034, 307.37669894645785, 132.0033784079643, 28.407665443332604]
    targets = [math.sin(math.radians(sample)) for sample in samples]

    X = np.array(samples)/360.0
    X = X.reshape(1, -1)
    y = np.array(targets).reshape(1, -1)

    weights_1 = np.random.randn(10, 1) * 0.1
    bias_1 = np.zeros(10)

    weights_2 = np.random.randn(1, 10) * 0.1
    bias_2 = np.zeros(1)

    l1 = Layer(1, weights_1, 10, sigmoid, bias_1)
    l2 = Layer(10, weights_2, 1, linear, bias_2)

    ann = NeuralNetwork([l1,l2])
    loss, output_pred = ann.run(
            X,
            y,
            0.25,
            sigmoid_derivative,
            linear_derivative
        )
    print("SINE EXAMPLE:\n")
    print(f"loss: {loss}")
    print(f"samples:\n{samples}")
    print(f"targets:\n{targets}")
    print(f"output_pred:\n{output_pred}")


main()