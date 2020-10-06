from .core import Op

import numpy as np

"""
This module contains a selection of useful Ops with forward and backward implementations.
"""

class Add(Op):
    """
    Op for element-wise matrix addition.
    """
    @staticmethod
    def forward(context, a, b):
        assert a.shape == b.shape, f'Arrays not the same sizes ({a.shape} {b.shape}).'
        return a + b

    @staticmethod
    def backward(context, go):
        return go, go

class Multiply(Op):
    """
    Op for element-wise matrix multiplication.
    """
    @staticmethod
    def forward(context, a, b):
        assert a.shape == b.shape, f'Arrays not the same sizes ({a.shape} {b.shape}).'

        context['a'] = a
        context['b'] = b

        return a * b

    @staticmethod
    def backward(context, go):
        a, b = context['a'], context['b']

        return go * b, go * a
        # -- note the reversal: the local gradient wrt a is b and the local graident wrt b is a.

class Log(Op):
    """
    Op for natural logarithm
    """
    @staticmethod
    def forward(context, x):

        context['x'] = x

        return np.log(x)

    @staticmethod
    def backward(context, go):

        x = context['x']

        return go / x

class Sum(Op):
    """
    Op that sums all elements in a tensor together, returning a scalar value.
    """

    @staticmethod
    def forward(context, x):

        context['xsize'] = x.shape
        return np.asarray(x.sum())

    @staticmethod
    def backward(context, go):
        assert go.shape == () # go should be scalar

        xsize = context['xsize']

        return np.full(shape=xsize, fill_value=go)

class MatrixMultiply(Op):
    """
    Op for matrix multiplication.
    """
    @staticmethod
    def forward(context, a, b):
        context['a'] = a
        context['b'] = b

        return np.matmul(a, b)

    @staticmethod
    def backward(context, go):
        a, b = context['a'], context['b']

        return np.matmul(go, b.T), np.matmul(a.T, go)

class BatchMM(Op):
    """
    Op for batched matrix/vector multiplication. Assumes a single matrix and a batch of vectors.

    -- In pytorch we would accomplish this with clever use of batchmm() and some squeezing/unsqueezing of dimensions,
       which is much more flexible, but for our current purposes, this is all we need.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(context, matrix, vectors):

        assert len(matrix.shape) == 2 and len(vectors.shape) == 2

        context['matrix'] = matrix
        context['vectors'] = vectors

        return np.matmul(vectors, matrix.T)

    @staticmethod
    def backward(context, go):
        matrix, vectors = context['matrix'], context['vectors']

        return  np.matmul(go.T, vectors), np.matmul(go, matrix)

class Sigmoid(Op):
    """
    Op for element-wise application of sigmoid function
    """

    @staticmethod
    def forward(context, input):

        sigx =  1 / (1 + np.exp(-input))
        context['sigx'] = sigx # store the sigmoid of x for the backward pass
        return sigx

    @staticmethod
    def backward(context, goutput):
        sigx = context['sigx'] # retrieve the sigmoid of x
        return goutput * sigx * (1 - sigx)

class Expand(Op):
    """
    Expand a singleton dimension in the input (that is, repeat the input a number of times along a given dimension of
    size 1, as is done in broadcasting).
    """

    @staticmethod
    def forward(context, input, *, dim, repeats):
        context['dim'] = dim

        assert input.shape[dim] == 1, 'Can only expand singleton dimensions'

        return np.repeat(input, repeats, axis=dim)

    @staticmethod
    def backward(context, goutput):
        dim = context['dim']

        return goutput.sum(axis=dim, keepdims=True)

class Softmax(Op):
    """
    Op for softmax operation on a
    """

    @staticmethod
    def forward(context, input):

        batch_size, num_classes = input.shape



        expd  = np.exp(input) # exponentiate
        normd = expd / expd.sum(axis=1, keepdims=True) # normalize

        context['normd'], context['expd'] = normd, expd

        return normd


    @staticmethod
    def backward(context, goutput):

        normd, expd = context['normd'], context['expd']

        return goutput * expd + expd * (goutput * expd).sum(axis=1, keepdims=True)

class Select(Op):
    """
    Select a single element from each row of a matrix. The target indices are given by an integer vector of the same
    length as the input matrix has rows.

    In pytorch, this operation would be accomplished with the (much more flexible) gather() function.
    """
    @staticmethod
    def forward(context, input, *, indices):
        """
        Note that indices and dim are not nodes in the graph, so we do not produce a gradient over them.

        :param context:
        :param input:
        :param indices: An integer vector of indices
        :return:
        """
        assert len(input.shape) == 2
        assert len(indices.shape) == 1
        assert indices.shape[0] == input.shape[0]

        context['input_size'] = input.shape

        indices = indices[:, None]
        context['indices'] = indices

        return np.take_along_axis(input, indices=indices, axis=1).squeeze()

    @staticmethod
    def backward(context, goutput):

        input_size = context['input_size']
        res = np.zeros(input_size) # everything that was not selected has gradient 0 (it did not affect the loss)

        indices = context['indices']
        np.put_along_axis(res, indices=indices, values=goutput[:, None], axis=1)

        return res

## The Ops below this line

class Squeeze(Op):
    """
    Remove a specific singleton dimensions
    """

    @staticmethod
    def forward(context, input, dim=0):
        context['dim'] = dim

        return input.squeeze(dim)

    @staticmethod
    def backward(context, goutput):
        dim = context['dim']

        return np.expand_dims(goutput, axis=dim)

class Unsqueeze(Op):
    """
    Insert a specific singleton dimensions
    """

    @staticmethod
    def forward(context, input, dim=0):
        context['dim'] = dim

        return np.expand_dims(input, axis=dim)

    @staticmethod
    def backward(context, goutput):
        dim = context['dim']

        return goutput.squeeze(dim)