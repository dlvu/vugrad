import sys
if sys.version_info < (3,6):
    sys.exit('A python version of 3.6 or higher is required. Consider creating a ptyhon virtual environment.')

from .core import TensorNode, OpNode, Module, Add, Multiply, MatrixMultiply
from .ops import *
from .modules import *
from .functions import load_synth, celoss, softmax