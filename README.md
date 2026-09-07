# ANN Fundamentals Lab

A small educational project implementing the fundamentals of an **Artificial Neural Network (ANN) from scratch using NumPy**.

The goal of this project is to understand how neural networks work internally rather than relying on machine learning frameworks such as TensorFlow or PyTorch.

## Features

- Fully connected neural network implementation
- Forward propagation
- Backpropagation
- Mean Squared Error (MSE) loss
- Sigmoid activation function
- Linear activation function
- Batch processing of multiple samples
- Gradient-based weight and bias updates
- Configurable network layers and activation functions

## Core Calculations

The neural network is implemented using the following fundamental calculations.

### Forward Propagation

For each layer:

Z = WX + b

A = f(Z)

where X is the input, W is the weight matrix, b is the bias, Z is the weighted sum, and f is the activation function.

Sigmoid Activation

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

Its derivative is:

$$
\sigma'(z) = \frac{e^{-z}}{(1 + e^{-z})^2}
$$

Linear Activation

$$
f(z) = z
$$

$$
f'(z) = 1
$$

The linear activation is used in the output layer for the sine regression experiment.

### Mean Squared Error

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
$$

Its derivative with respect to the predictions is:

$$
\frac{2}{N}(\hat{y}-y)
$$

### Backpropagation

For a layer:

$$
\frac{\partial L}{\partial A}
\odot f'(Z)
$$

The weight and bias gradients are:

$$
\frac{\partial L}{\partial Z}X^T
$$

$$
\sum \frac{\partial L}{\partial Z}
$$

For a hidden layer:

$$
\left(
W_{l+1}^T
\frac{\partial L}{\partial Z_{l+1}}
\right)
\odot f'(Z_l)
$$

Parameter Update

Gradient descent is used to update the weights and biases:

$$
W \leftarrow W-\eta\frac{\partial L}{\partial W}
$$

$$
b \leftarrow b-\eta\frac{\partial L}{\partial b}
$$

where $\eta$ is the learning rate.

## Experiments

The project currently includes two simple experiments:

### XOR Classification

A small neural network is trained to learn the XOR function.

**Architecture:**

```text
2 inputs → 2 hidden neurons → 1 output
```

This experiment demonstrates how a multi-layer network can learn a non-linear relationship.

### Sine Regression

A neural network is trained to approximate the sine function using randomly generated angles.

**Architecture:**

```text
1 input → 10 hidden neurons → 1 output
```

The hidden layer uses the sigmoid activation function, while the output layer uses a linear activation for regression.

## Project Structure

```text
ann-fundamentals-lab/
├── src/
│   ├── layer.py
│   └── neuralnetwork.py
├── main.py
├── README.md
└── requirements.txt
```

## Requirements

- Python 3.x
- NumPy

Install the dependency with:

```bash
pip install -r requirements.txt
```

## Running the Project

Run the experiments from the project root:

```bash
python main.py
```

The program trains the networks and prints the training loss and predictions.

## Purpose

This project is primarily a **learning exercise** focused on understanding the mathematical and computational foundations of neural networks, including matrix operations, activation functions, gradients, and backpropagation.
