from _context import vugrad

import numpy as np

# for running from the command line
from argparse import ArgumentParser

import vugrad as vg

# Parse command line arguments
parser = ArgumentParser()

parser.add_argument('-D', '--dataset',
                dest='data',
                help='Which dataset to use. [synth, mnist]',
                default='synth', type=str)

parser.add_argument('-b', '--batch-size',
                dest='batch_size',
                help='The batch size (how many instances to use for a single forward/backward pass).',
                default=128, type=int)

parser.add_argument('-e', '--epochs',
                dest='epochs',
                help='The number of epochs (complete passes over the complete training data).',
                default=20, type=int)

parser.add_argument('-l', '--learning-rate',
                dest='lr',
                help='The learning rate. That is, a scalar that determines the size of the steps taken by the '
                     'gradient descent algorithm. 0.1 works well for synth, 0.0001 works well for MNIST.',
                default=0.01, type=float)

args = parser.parse_args()

## Load the data
if args.data == 'synth':
    (xtrain, ytrain), (xval, yval), num_classes = vg.load_synth()
elif args.data == 'mnist':
    (xtrain, ytrain), (xval, yval), num_classes = vg.load_mnist(final=False, flatten=True)
else:
    raise Exception(f'Dataset {args.data} not recognized.')

print(f'## loaded data:')
print(f'         number of instances: {xtrain.shape[0]} in training, {xval.shape[0]} in validation')
print(f' training class distribution: {np.bincount(ytrain)}')
print(f'     val. class distribution: {np.bincount(yval)}')

num_instances, num_features = xtrain.shape

## Create the model.
mlp = vg.MLP(input_size=num_features, output_size=num_classes)

n, m = xtrain.shape
b = args.batch_size

print('\n## Starting training')

cl = '...'

for epoch in range(args.epochs):

    print(f'epoch {epoch:03}')

    if epoch % 1 == 0:
        ## Compute validation accuracy

        o = mlp(vg.TensorNode(xval))
        oval = o.value

        predictions = np.argmax(oval, axis=1)
        num_correct = (predictions == yval).sum()
        acc = num_correct / yval.shape[0]

        o.clear() # gc the computation graph

        print(f'       accuracy: {acc:.4}')

    cl = 0.0 # running sum of the training loss

    # We loop over the data in batches of size `b`
    for fr in range(0, n, b):

        # The end index of the batch
        to = min(fr + b, n)

        # Slice out the batch and its corresponding target values
        batch, targets = xtrain[fr:to, :], ytrain[fr:to]

        # Wrap the inputs in a Node
        batch = vg.TensorNode(value=batch)

        outputs = mlp(batch)
        loss = vg.celoss(outputs, targets)
        # -- The computation graph is now complete. It consists of the mlp, together with the computation of
        #    the scalar loss.
        # -- The variable `loss` is the TreeNode at the very top of our computation graph. This means we can call
        #    it to perform operations on the computation graph, like clearing the gradients, starting the backpropgation
        #    and clearing the graph.

        cl += loss.value
        # -- We must be careful here to extract the _raw_ value for the running loss. What would happen if we kept
        #    a running sum using the TensorNode?

        # Start the backpropagation
        loss.backward()

        # pply gradient descent
        for parm in mlp.parameters():
            parm.value -= args.lr * parm.grad
            # -- Note that we are directly manipulating the members of the parm TensorNode. This means that for this
            #    part, we are not building up a computation graph.

        # -- In Pytorch, the gradient descent is abstracted away into an Optimizer. This allows us to build slightly more
        #    complexoptimizers than plain graident descent.

        # Finally, we need to reset the gradients to zero ...
        loss.zero_grad()
        # ... and delete the parts of the computation graph we don't need to remember.
        loss.clear()

    print(f'   running loss: {cl:.4}')
