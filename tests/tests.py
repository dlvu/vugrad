import _context

import unittest
import torch
from torch import nn

import vugrad as vg
import numpy as np

"""
This is mostly a collection of test code at the moment, rather than a proper suite of unit tests.

"""

def fd_mlp():
    """
    Test the framework by computing finite differences approximation to the gradient for a random parameter

    :return:
    """
    IDX = (0, 0)

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

    eps = max(1.5e-8 * parm.value[IDX], 10e-12)
    parm.value[IDX] += eps

    outputs1 = mlp(batch)
    loss1 = vg.celoss(outputs1, targets)

    fd_deriv = (loss1.value - loss0.value) / eps

    print(f'finite differences: {fd_deriv:.3}')
    print(f'          backprop: {bp_deriv:.3}')

def finite_differences(function, input='eye'):
    """
    Test the framework by computing finite differences approximation to the gradient.

    :param function: Some function that takes a matrix and returns a scalar (using Ops).
    :return:
    """

    N = 5

    if type(input) == str:
        if input == 'eye':
            inp = vg.TensorNode(np.eye(N))
        elif input == 'rand':
            inp = vg.TensorNode(np.random.randn(N, N))
        else:
            raise Exception()
    else:
        inp = vg.TensorNode(input)

    N, M = inp.size()

    for i in range(N):
        for j in range (M):

            eps = max(1.5e-8 * inp.value[i, j], 10e-12)
            # -- This is supposedly a good epsilon value to use.

            loss0 = function(inp)
            loss0.backward()

            bp_deriv = inp.grad[i, j]

            inpe = vg.TensorNode(inp.value.copy())
            inpe.value[i, j] += eps

            loss1 = function(inpe)

            fd_deriv = (loss1.value - loss0.value) / eps

            print(i, j)
            print(f'  finite differences: {fd_deriv:.3}')
            print(f'            backprop: {bp_deriv:.3}')

            loss0.zero_grad()
            loss1.zero_grad()
            loss0.clear()
            loss1.clear()

class TestUtil(unittest.TestCase):
    """

    """

    def test_fd0(self):
        """
        Test the backprop using finite differences
        :return:
        """

        finite_differences(lambda x : vg.Sum.do_forward(x))


    def test_fd1(self):
        """
        Test the backprop using finite differences
        :return:
        """

        finite_differences(lambda x : vg.Sum.do_forward(vg.Sigmoid.do_forward(x)), input='rand')

    def test_fd2(self):
        """
        Test the backprop using finite differences
        :return:
        """

        finite_differences(input='rand', function=lambda x:
            vg.Sum.do_forward(
                vg.Sigmoid.do_forward(
                    vg.MatrixMultiply.do_forward(x, x)
                )))

    def test_fd3(self):
        """
        Test the backprop using finite differences
        :return:
        """
        def fn(x):

            x = vg.Exp.do_forward(x)
            x = vg.Normalize.do_forward(x)

            return vg.Sum.do_forward(x)

        finite_differences(
            # input=np.asarray([[10.2, 20.4]]),
            input=np.asarray([[0.6931471805599453, 0.0]]),
            # input=np.random.randn(10, 2),
            function=fn)

    def test_mlp(self):

        fd_mlp()