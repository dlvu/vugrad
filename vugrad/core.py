import numpy as np

"""
This module contains the core components of the autodiff system: the two types of nodes that make up the computation graph
(TensorNodes and OpNodes) and the class definition of an Op.

The main algorithm of backpropagation is implementated recursively in the backward functions of the nodes.
"""

class TensorNode:
    """
    Represents a value node in the computation graph: a tensor value linked to its
    the computational history
    """

    def __init__(self, value : np.ndarray, source=None):

        self.value = value   # the raw value that this node holds
        self.source = source # the OpNode that produced this Node

        self.grad = np.zeros(shape=value.shape) # This is where we will put the gradient when we compute the loss

        # -- The gradient is defined as a tensor with the same dimensions as the value. At element `index` it contains
        #    the derivative of the loss with respect to the parameter `value[index]`.

    def size(self, dim=None):
        """
        Return a tuple representing the dimensions of this tensor, or the specific size of one dimension.
        :param dim: If None, return sizes of all dimensions. Otherwise, return the size of dimension ind
        :return:
        """
        if dim == None:
            return self.value.shape

        return self.value.shape[dim]

    def zero_grad(self):
        """
        Set the gradient to zero for this node and any below it.
        :return:
        """


        self.grad.fill(0.0)

        if self.source is not None:
            self.source.zero_grad()

    def clear(self):
        """
        Disconnects the computation graph. We simply remove the connections between the Tensor nodes and the Op nodes.
        By doing this, anything that is not referenced by anything else gets removed by the garbage collector. The nodes
        that represent parameters are still connected to their modules, so they won't be cleared.

        :return:
        """

        if self.source is not None:
            self.source.clear()

        self.source = None

    ## For common ops, we add utility methods to the Node object

    def __add__(self, other):
        if type(other) == float:
            other = TensorNode(np.asarray(other))

        return Add.do_forward(self, other)

    def __mul__(self, other):
        if type(other) == float:
            other = TensorNode(np.asarray(other))

        return Multiply.do_forward(self, other)

    def matmul(self, other):
        return MatrixMultiply.do_forward(self, other)

    def backward(self, start=True):
        """
        Start the backpropagation from this node. This will fail if the node holds anything other than a scalar value.

        :return:
        """
        # print(f'Backward in TensorNode with shape {self.size()} and source op {None if self.source is None else self.source.op}.')

        if start:
            if  self.value.squeeze().shape != ():
                raise Exception('backward() can only start from a scalar node.')

            self.grad = np.ones_like(self.value)

        # -- the gradient of the loss node is 1, with the same shape as the loss node itself

        if self.source is not None:
            self.source.backward()

    def __str__(self):
        return f'TensorNode[size {self.size()}, source {self.source.op if self.source is not None else None}].'

class OpNode:
    """
    A particular instance of an op applied to some inputs

    (i.e. a diamond node in a computation graph)
    """

    def __init__(self, op, context, inputs):
        super().__init__()

        self.op = op
        self.context = context
        self.inputs = inputs

        self.outputs = None

    def backward(self):
        """
         Walk backwards down the graph to compute the gradients.

         Note that this should be a breadth-first walk. When we get to a particular op, we need to be sure that all its
         outputs have already been visited.
         """
        # print(f'Backward of op {self.op}.')

        # extract the gradients over the outputs (these have been computed already)
        goutputs_raw = [output.grad for output in self.outputs]

        # compute the gradients over the inputs
        ginputs_raw = self.op.backward(self.context, *goutputs_raw)

        if not type(ginputs_raw) == tuple:
            ginputs_raw = (ginputs_raw,)

        # store the computed gradients in the input nodes
        for node, grad in zip(self.inputs, ginputs_raw):
            assert node.grad.shape == grad.shape, f'node shape is {node.size()} but grad shape is {grad.shape}'

            node.grad += grad
            # -- Note that we add the gradient to the one already there. This means that for TensorNodes that are the
            #    input to two ops, we are automatically implementing the multivariate chain rule. Every op adds its part
            #    of the gradient to the .grad part of its inputs.

        for node in self.inputs:
            node.backward(start=False)


    def zero_grad(self):
        """
        Set the gradient to zero all inputs and their ancestors.
        :return:
        """
        for node in self.inputs:
            node.zero_grad()

    def clear(self):
        """
        Disconnects this OpNode form the computation graph, so that it will be garbage collected when nothing refers to
        it.
        :return:
        """

        for node in self.inputs:
            node.clear()

        self.inputs = None
        self.outputs = None

        #-- The garbage collector should handle the rest.

class Op:
    """
    Abstract class to represent an operation.

    Note that this is a class with only static methods. We never instantiate it, we just use it to bundle
    the forward and backward methods in a way that we can refer to the op as a whole.

    """

    @classmethod
    def do_forward(cls, *inputs, **kwargs):
        """
        Apply the op to a tuple of tensor nodes. This method unwraps the values, prepares a context and runs the
        computation on the raw values.

        This method takes care of all the graph construction and defers to forward() for the actual computation of the
        output tensor given the inputs.

        :param inputs: The TreeNodes that serve as inputs to the op. All non-keyword arguments are taken as TreeNode
                       inputs
        :param kwargs: Any key word arguments are taken as non-TreeNode constants. For these, no gradient will be computed.
        :return:
        """
        context = {}

        assert all([type(input) == TensorNode for input in inputs]), 'All inputs to an Op should be TensorNodes.'

        # extract the raw input values
        inputs_raw = [input.value for input in inputs]

        # compute the raw output values
        outputs_raw = cls.forward(context, *inputs_raw, **kwargs)

        if not type(outputs_raw) == tuple:
            outputs_raw = (outputs_raw, )

        opnode = OpNode(cls, context, inputs)

        outputs = [TensorNode(value=output, source=opnode) for output in outputs_raw]
        opnode.outputs = outputs

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    @staticmethod
    def forward(context : dict, *inputs, **kawrgs):
        """
        Compute the output given the inputs.

        :param inputs: A tuple of input arrays. Note that these are "unwrapped" raw numpy arrays.
        :param kwargs: Any constant values. These are not nodes, and no gradient should be computed for them.
        :param context: A dictionary where we can store anything we may need to remember for the backward pass.
        :return:
        """
        pass

    @staticmethod
    def backward(context : dict, *goutput):
        """
        Compute the gradient of the loss over the inputs given the loss over the outputs

        NB: note that backward does _not_ compute just the local derivative: it's the local
        derivative times the upstream gradient.

        :param goutputs: A tuple of gradients over the output nodes. Note that these are "unwrapped" raw numpy arrays.
        :param context: The context dictionary from the forward pass
        :return:
        """
        pass

class Module:
    """
    A _module_ combines some parameter tensors together with a method for
    computing some function from them (using various ops)

    Our module doesn't do anything more than serve as a superclass. In Pytorch the Module class
    does a lot more, like automatically registering parameters.

    Note that modules have a forward(), but they _don't_ have a backward(). The forward just combines a
    bunch of Ops which each have their own backward, so we can let the autodiff system take care of working
    out the backward.
    """
    def __init__(self):
        pass

    def forward(self):
        pass

    # -- We alias the forward() method with the magic function __call__ this means that our module instance becomes
    #    callable, like a function. For instance if we make an MLP with `mlp = MLP(...)`, we can then call its
    #    forward function with `mlp(input)`.
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self):
        """
        Return the parameters of this module. These are the nodes that we will keep between training runs, and that
        will be updated during learning.

        :return: A list of TensorNode objects
        """
        pass

## The most basic Ops. These are defined here so we can make utility functions for them in the TensorNode class. The
#  rest are defined in ops.py

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
        # -- note the reversal: the local gradient wrt a is b and the local gradient wrt b is a.


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
