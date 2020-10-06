import _context

import unittest
import torch
from torch import nn

import vugrad as vg
import numpy as np

def fd_mlp():
    """
    Test the framework by computing finite differences approximation to the gradient for a random parameter

    :return:
    """
    IDX = (0, 0)
    EPS = 10-5

    (xtrain, ytrain), (xval, yval), num_classes = vg.load_synth()

    # Slice out a batch and its corresponding target values
    batch, targets = xtrain[:100, :], ytrain[:100]

    # Wrap the inputs in a Node
    batch = vg.TensorNode(value=batch)

    num_instances, num_features = xtrain.shape
    mlp = vg.MLP(input_size=num_features, output_size=num_classes)

    parm = mlp.parameters()[0]

    outputs0 = mlp(batch)
    loss0 = vg.celoss(outputs0, targets)

    loss0.backward()

    bp_deriv = parm.grad[IDX]

    parm.value[IDX] += EPS

    outputs1 = mlp(batch)
    loss1 = vg.celoss(outputs1, targets)

    fd_deriv = (loss1.value - loss0.value) / EPS

    print(f'finite differences: {fd_deriv:.3}')
    print(f'          backprop: {bp_deriv:.3}')

def finite_differences(function):
    """
    Test the framework by computing finite differences approximation to the gradient for a random parameter

    :param function: Some function that takes a matrix and returns a scalar (using Ops).
    :return:
    """

    N = 5
    for i in range(N):
        for j in range (N):

            idx = (i, j)

            eye = vg.TensorNode(np.eye(N) * 5)

            eps = max(1.5e-8 * eye.value[idx], 10e-12)

            loss0 = function(eye)
            loss0.backward()

            bp_deriv = eye.grad[idx]

            eye.value[idx] += eps

            loss1 = function(eye)

            fd_deriv = (loss1.value - loss0.value) / eps

            print(i, j)
            print(f'  finite differences: {fd_deriv:.3}')
            print(f'            backprop: {bp_deriv:.3}')

class TestUtil(unittest.TestCase):
    """

    """

    def test_fd1(self):
        """
        Test the backprop using finite differences
        :return:
        """

        finite_differences(lambda x : vg.Sum.do_forward(vg.Sigmoid.do_forward(x)))
