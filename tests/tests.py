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

    def testmax(self):

        x = np.asarray([[0., 1.],[4., 5.],[9, 0.]])

        ctx = {}
        vg.RowMax.forward(ctx, x)
        grad = vg.RowMax.backward(ctx, np.asarray([.1, .2, .3]))

        self.assertTrue( (np.asarray([[0.,  .1], [0., .2 ], [.3, 0. ]]) == grad ).all() )

    def testsum(self):

        x = np.asarray([[0., 1.],[4., 0.],[9, 0.]])

        ctx = {}
        vg.RowMax.forward(ctx, x)
        grad = vg.RowSum.backward(ctx, np.arange(3.0) + 0.1)

        self.assertTrue( (np.asarray([[0.1,  0.1], [1.1, 1.1], [2.1, 2.1]]) == grad ).all() )

    def testlogsoftmax(self):

        x = np.asarray([[0., 0.],[2., 0.],[3., 0.]])

        x = vg.TensorNode(x)

        s = np.exp(vg.logsoftmax(x).value).sum(axis=1)
        self.assertTrue( ( (s - 1.0) ** 2. < 1e-10).all() )

    def testlogsoftmax2(self):

        x = np.random.randn(4, 5)
        x = vg.TensorNode(x)

        els = np.exp(vg.logsoftmax(x).value)
        s = vg.softmax(x).value
        self.assertTrue( ((els - s) ** 2. < 1e-7).all() )

    def testdiamond(self):

        a = vg.TensorNode(np.asarray([1.0]))
        b = vg.Id.do_forward(a)
        c1, c2 = vg.Id.do_forward(b), vg.Id.do_forward(b)
        d = c1 + c2

        a.name  = 'a'
        b.name  = 'b'
        c1.name = 'c1'
        c2.name = 'c2'
        d.name  = 'd'

        # a.debug = True

        d.backward()
        self.assertEqual(2.0, float(a.grad))

    def testdoublediamond(self):

        a0 = vg.TensorNode(np.asarray([1.0]))
        a = vg.Id.do_forward(a0)
        b1, b2 = vg.Id.do_forward(a), vg.Id.do_forward(a)
        c1, c2, c3, c4 = vg.Id.do_forward(b1), vg.Id.do_forward(b1), vg.Id.do_forward(b2), vg.Id.do_forward(b2)
        d1 = c1 + c2
        d2 = c3 + c4
        e = d1 + d2

        e.backward()
        self.assertEqual(4.0, float(a.grad))

    def testseqdiamond(self):

        a = vg.TensorNode(np.asarray([1.0]))
        b = vg.Id.do_forward(a)
        c1, c2 = vg.Id.do_forward(b), vg.Id.do_forward(b)
        d = c1 + c2
        e = vg.Id.do_forward(d)
        f = e + a
        g1, g2 = vg.Id.do_forward(f), vg.Id.do_forward(f)
        h = g1 + g2

        h.backward()
        self.assertEqual(2.0, float(f.grad))
        self.assertEqual(2.0, float(e.grad))
        self.assertEqual(2.0, float(c1.grad))
        self.assertEqual(4.0, float(b.grad))
        self.assertEqual(6.0, float(a.grad))