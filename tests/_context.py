"""
This is a little bit of python trickery to allow you to run the test without installing the vugrad package.
Ignore this.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../vugrad')))

import vugrad