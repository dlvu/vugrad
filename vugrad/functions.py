from .ops import Log, Select, Sum, Normalize, Exp, Sigmoid, RowMax, Expand, RowSum, Unsqueeze, Id
from .mnist import init, load
from .core import TensorNode

import numpy as np
import os

"""
A set of utility functions
"""

def load_synth(num_train=60_000, num_val=10_000):
    """
    Load some very basic synthetic data that should be easy to classify. Two features, so that we can plot the
    decision boundary (which is an ellipse in the feature space).

    :param num_train: Number of training instances
    :param num_val: Number of test/validation instances
    :param num_features: Number of features per instance
    :return: Two tuples (xtrain, ytrain), (xval, yval) the training data is a floating point numpy array:
    """

    THRESHOLD = 0.6
    quad = np.asarray([[1, 0.5], [1, .2]])

    ntotal = num_train + num_val

    x = np.random.randn(ntotal, 2)

    # compute the quadratic form
    q = np.einsum('bf, fk, bk -> b', x, quad, x)
    y = (q > THRESHOLD).astype(np.int)

    return (x[:num_train, :], y[:num_train]), (x[num_train:, :], y[num_train:]), 2


def load_mnist(final=False, flatten=True):
    """
    Load the MNIST data

    :param final: If true, return the canonical test/train split. If false, split some validation data from the training
       data and keep the test data hidden.
    :param flatten:
    :return:
    """

    if not os.path.isfile('mnist.pkl'):
        init()

    xtrain, ytrain, xtest, ytest = load()
    xtl, xsl = xtrain.shape[0], xtest.shape[0]

    if flatten:
        xtrain = xtrain.reshape(xtl, -1)
        xtest  = xtest.reshape(xsl, -1)

    if not final: # return the flattened images
        return (xtrain[:-5000], ytrain[:-5000]), (xtrain[-5000:], ytrain[-5000:]), 10

    return (xtrain, ytrain), (xtest, ytest), 10

def celoss(outputs, targets):
    """
    The cross-entropy loss as explained in the slides.

    We could implement this as an op, if we wanted to (we would just need to work out the backward). However, in this
    case we've decided to be lazy and implement it as a basic python function. Notice that by simply making computations,
    the code is building the final parts of our computation graph.

    This could also be implemented as a Module (as it is in the pytorch tutorial), but that doesn't add much, since this
    part of our computation graph has no parameters to store.

    :param outputs: Predictions from the model, a distribution over the classes
    :param targets: True class values, given as integers
    :return: A single loss value: the lower the value, the better the outputs match the targets.
    """

    logprobs = Log.do_forward(outputs)

    return logceloss(logprobs, targets)

def logceloss(logprobs, targets):
    """
    Implementation of the cross-entropy loss from logprobabilities

    We separate this from the celoss, because computing the probabilities explicitly (as done there) is numerically
    unstable. It's much more stable to compute the log-probabilities directly, using the log-softmax function.

    :param outputs:
    :param targets:
    :return:
    """

    # The log probability of the correct class, per instance
    per_instance = Select.do_forward(logprobs, indices=targets)

    # The loss sums all these. The higher the better, so we return the negative of this.
    return Sum.do_forward(per_instance) * - 1.0

def sigmoid(x):
    """
    Wrap the sigmoid op in a funciton (just for symmetry with the softmax).

    :param x:
    :return:
    """
    return Sigmoid.do_forward(x)

def softmax(x):
    """
    Applies a row-wise softmax to a matrix

    NB: Softmax is almost never computed like this in serious settings. It's much better
        to start from logits and use the logsumexp trick, returning
        `log(softmax(x))`. See the logsoftmax function below.

    :param x: A matrix.
    :return: A matrix of the same size as x, with normalized rows.
    """

    return Normalize.do_forward(Exp.do_forward(x))

def logsoftmax(x):
    """
    Computes the logarithm of the softmax.

    This function uses the "log sum exp trick" to compute the logarithm of the softmax
    in a numerically stable fashion.

    Here is a good explanation: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

    :param x: A matrix.
    :return: A matrix of the same size as x, with normalized rows.
    """

    # -- Max over the rows and expand back to the size of x

    xcols = x.value.shape[1]

    xmax = RowMax.do_forward(x)
    xmax = Unsqueeze.do_forward(xmax, dim=1)
    xmax = Expand.do_forward(xmax, repeats=xcols, dim=1)

    assert(xmax.value.shape == x.value.shape), f'{xmax.value.shape}    {x.value.shape}'

    diff = x - xmax

    denominator = RowSum.do_forward( Exp.do_forward(diff) )
    denominator = Log.do_forward(denominator)

    denominator = Unsqueeze.do_forward(denominator, dim=1)
    denominator = Expand.do_forward(denominator, repeats=xcols, dim=1)

    assert(denominator.value.shape == x.value.shape), f'{denominator.value.shape}    {x.value.shape}'

    res = diff - denominator

    return res
