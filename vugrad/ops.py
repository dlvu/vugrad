from .core import Op, TensorNode

import numpy as np

"""
This module contains a selection of useful Ops with forward and backward implementations.

Note that three ops (Add, Multiply, MatrixMultiply) are implemented in the core module to avoid circular imports.

"""


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


class Id(Op):
    """
    Identity operation (useful for debugging as it does add a new node to the graph)
    """
    @staticmethod
    def forward(context, x):

        return x

    @staticmethod
    def backward(context, go):

        return go

class Exp(Op):
    """
    Op for natural exponent
    """
    @staticmethod
    def forward(context, x):

        context['expd'] = np.exp(x)
        return context['expd']

    @staticmethod
    def backward(context, go):

        return go * context['expd']

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

class RowMax(Op):
    """
    Op that takes the row-wise maximum of a matrix, resulting in a vector
    """

    @staticmethod
    def forward(context, x):

        context['xshape'] = x.shape
        context['idxs'] = np.argmax(x, axis=1) # -- indices of maximum elements

        return np.amax(x, axis=1)

    @staticmethod
    def backward(context, go):
        xs = context['xshape']
        idxs = context['idxs']

        # The gradient for the output is a zero matrix with the upstream gradients
        # at the positions of the row-wise maxima of x
        z = np.zeros(shape=xs)
        z[np.arange(go.shape[0]), idxs] = go

        return z

class RowSum(Op):
    """
    Op that takes the row-wise sum of a matrix, resulting in a vector
    """

    @staticmethod
    def forward(context, x):

        context['xshape'] = x.shape

        return np.sum(x, axis=1)

    @staticmethod
    def backward(context, go):
        assert len(go.shape) == 1

        xshape = context['xshape']
        gx = np.broadcast_to(go.copy()[:, None], shape=xshape)
        # gx = np.repeat(go[:, None], axis=1, repeats=xshape[1])

        return gx

class Normalize(Op):
    """
    Op that normalizes a matrix along the rows
    """
    @staticmethod
    def forward(context, x):

        sumd = x.sum(axis=1, keepdims=True)

        context['x'], context['sumd'] = x, sumd

        return x / sumd

    @staticmethod
    def backward(context, go):

        x, sumd = context['x'], context['sumd']

        return (go / sumd) - ((go * x)/(sumd * sumd)).sum(axis=1, keepdims=True)

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
        assert input.shape[dim] == 1, 'Can only expand singleton dimensions'

        context['dim'] = dim

        return np.repeat(input, repeats, axis=dim)

    @staticmethod
    def backward(context, goutput):
        dim = context['dim']

        return goutput.sum(axis=dim, keepdims=True)

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

## The Ops below this line aren't used in the course exercises. You may ignore them.

class Squeeze(Op):
    """
    Remove a specific singleton dimension
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