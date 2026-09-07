from src.loss import mse_derivative,mse
from src.activation import sigmoid_derivative
from src.layer import *
import numpy as np
class NeuralNetwork:

    def __init__(self,layers):
        self._layers = layers

    @property
    def layers(self):
        return self._layers.copy()

    def forward(self,inputs,output_act):
        output=inputs
        for i in self._layers:
            output = i.calc_activation(output)
        loss = mse(output_act,output)
        return output,loss

    def backpropagation(self,output_act,output_pred,learning_rate,activation_derivative_hidden,activation_derivative_output):
        layer = self.layers[-1]
        grad_z = mse_derivative(output_act,output_pred) * activation_derivative_output(layer.z)
        grad_w = (grad_z @ layer.inputs.T)
        grad_b = (np.sum(grad_z, axis=1, keepdims=True))
        new_weights = layer.weights - learning_rate*grad_w
        new_bias = layer.bias - learning_rate*grad_b.reshape(-1)

        layer.store_new_weights_and_bias(new_weights,new_bias)

        for i in range(len(self.layers)-2,-1,-1):
            layer = self.layers[i]
            prev_layer = self.layers[i+1]
            grad_z = (prev_layer.weights.T @ grad_z) * activation_derivative_hidden(layer.z)
            grad_w = grad_z @ layer.inputs.T
            grad_b = (np.sum(grad_z, axis=1, keepdims=True))

            new_weights = layer.weights - learning_rate * grad_w
            new_bias = layer.bias - learning_rate*grad_b.reshape(-1)

            layer.store_new_weights_and_bias(new_weights,new_bias)


        for layer in self.layers:
            layer.update_weights_and_bias()

        return grad_w,grad_b

    def run(self,inputs,output_act,learning_rate,activation_derivative_hidden,activation_derivative_output,max_epochs=10000,):
        loss = float("inf")
        epoch=0
        #each epoch
        while loss > 0.001 and epoch < max_epochs:

            output_pred,loss = self.forward(inputs,output_act)
            grad_w,grad_b = self.backpropagation(output_act,output_pred,learning_rate,activation_derivative_hidden,activation_derivative_output)


            epoch=epoch+1
            #print(f"\ncurrent epoch: {epoch} - current loss: {loss}")

        output_pred, loss = self.forward(inputs,output_act)
        print(f"\ncurrent epoch: {epoch} - current loss: {loss}")

        return loss,output_pred
