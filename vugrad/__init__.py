import sys
if sys.version_info < (3,8):
    sys.exit('A python version of 3.8 or higher is required. Consider creating a ptyhon virtual environment.')

from .core import TensorNode, OpNode, Module
from .ops import *
from .modules import *
from .functions import load_synth