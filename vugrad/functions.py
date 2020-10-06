from .ops import Log, Select, Sum
import numpy as np

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


def load_mnist(final=False):
    """
    Load the MNIST data

    :param final:
    :return:
    """

    pass

def celoss(outputs, targets):
    """
    The cross-entropy loss as explained in the slides.

    We could implement this as an op, if we wanted to (we would just need to work out the backward). However, in this
    case we've decided to be lazy and implement it as a basic python function. Notice that by simply making computations,
    the code is building the final parts of our computation graph.

    This could also be implemented as a Module (as it is in the pytorch tutorial), but that doesn't add much, since this
    part of our computation graph has no parameters to store.

    NB: This implementation is numerically unstable. It's much better to let the model outptu logits, and
        work with those. For our current purposes, however, this will do.

    :param outputs: Predictions from the model, a distribution over the classes
    :param targets: True class values, given as integers
    :return: A single loss value: the lower the value, the better the outputs match the targets.
    """

    logprobs = Log.do_forward(outputs)

    # The log probability of the correct class, per instance
    per_instance = Select.do_forward(logprobs, indices=targets)

    # the loss sums all these. The higher the better, so we return the negative of this.
    return Sum.do_forward(per_instance) * -1.0

    # -- To see how this loss derives from the entropy, consult the slides.