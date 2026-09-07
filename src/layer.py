class Layer:
    #input shape (3x1)
    #output shape (4x1)
    #weight shape (4x3)
    #bias shape (4x1)
    #formula = w @ i + b = o

    def __init__(self,num_inputs,weights,num_outputs,activation,bias):
        self._num_inputs = num_inputs
        self._num_outputs = num_outputs
        self._weights = weights
        self.bias = bias
        self.activation = activation

    @property
    def num_inputs(self):
        return self._num_inputs

    @property
    def num_outputs(self):
        return self._num_outputs

    @property
    def weights(self):
        return self._weights.copy()

    @property
    def new_weights(self):
        return self._new_weights.copy()

    @property
    def new_bias(self):
        return self._new_bias.copy()

    @property
    def z(self):
        return self._z.copy()

    @property
    def inputs(self):
        return self._inputs.copy()


    def set_weights(self,weights):
        if weights.shape == self._weights.shape:
            self._weights = weights
        else:
            raise ValueError("invalid number of weights")

    def store_new_weights_and_bias(self,weigths,bias):
        self._new_weights = weigths
        self._new_bias = bias

    def update_weights_and_bias(self):
        self.set_weights(self._new_weights)
        self.bias = self._new_bias

    def calc_activation(self,inputs):
        if inputs.shape[0] == self._num_inputs:
            self._inputs = inputs
            self._z = self._weights @ inputs + self.bias.reshape(-1,1)
            return self.activation(self._z)
        else:
            raise ValueError("invalid number of inputs")