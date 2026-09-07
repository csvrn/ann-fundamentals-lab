class Neuron:

    def __init__(self,weights,bias,activation):
        self.bias=bias
        self._weights=weights
        self.activation=activation

    @property
    def weights(self):
        return self._weights.copy()

    def set_weights(self,weights):
        if len(weights)==len(self._weights):
            self._weights=weights
        else:
            raise ValueError("invalid weights")

    def activation_output(self,inputs):
        if len(inputs)!=len(self.weights):
            raise ValueError("number of weights and inputs dont match")
        weighted_sum=0
        for index,input in enumerate(inputs):
            weighted_sum = weighted_sum + input*self._weights[index]
        self.output=self.activation(weighted_sum+self.bias)
        return self.output

