from .core import Module, TensorNode

from .ops import *
from .functions import *

class Linear(Module):
    """
    A linear operation. Applies a matrix transformation and a vector translation.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        # weights of the matrix transformation
        glorot_std = 2.0 / (input_size + output_size) # scalar for Glorot init
        w = np.random.randn(output_size, input_size) * glorot_std
        self.w = TensorNode(w)

        # weights of the bias (the translation)
        b = np.zeros((1, output_size))
        self.b = TensorNode(b)
        # -- We initialize the biases to zero for simplicity. This is a common approach, but with ReLU units it's
        #    sometimes best to add a little noise to avoid dead neurons.

    def forward(self, input):

        outsize, insize = self.w.size()
        n, f = input.size()

        assert f == insize, f'Number of features in input ({f}) does not match input dimension ({insize}).'
        assert len(input.size()) == 2

        # Multiply all input vectors by the weight matrix.
        x = BatchMM.do_forward(self.w, input)

        assert x.size() == (n, outsize)

        exb = Expand.do_forward(self.b, dim=0, repeats=n)
        # -- We are broadcasting the (1, outsize) vector b over the (n, outsize) matrix x. Numpy normally does this
        #    automatically, if we just do `x + self.b`, but we wouldn't get a gradient over that operation. Expand
        #    is a minimal broadcasting op that is sufficient for our purposes.
        # -- In pytorch, full-featured broadcasting is implemented so there you would actually be able to do `x + self.b`.

        assert x.size() == exb.size()

        return x + exb

    def parameters(self):
        return [self.w, self.b]

class MLP(Module):
    """
    A simple MLP with one hidden layer, and a sigmoid non-linearity on the hidden layer and a softmax on the
    output.
    """

    def __init__(self, input_size, output_size, hidden_mult=4):
        """

        :param input_size:
        :param output_size:
        :param hidden_mult: Multiplier that indicates how many times bigger the hidden layer is than the input layer.
        """
        super().__init__()

        hidden_size = hidden_mult * input_size
        # -- There is no common wisdom on how big the hidden size should be, apart from the idea
        #    that it should be strictly _bigger_ than the input. If it isn't, the network won't be
        #    able to learn closed shapes like circles.

        self.layer1 = Linear(input_size, hidden_size)
        self.layer2 = Linear(hidden_size, output_size)

    def forward(self, input):

        assert len(input.size()) == 2

        # first layer
        hidden = self.layer1(input)

        # non-linearity
        hidden = sigmoid(hidden)

        # second layer
        output = self.layer2(hidden)
        output = softmax(output)

        return output

    def parameters(self):

        return self.layer1.parameters() + self.layer2.parameters()
