import sys
import os

# Add lama directory to path so saicinpainting can be imported as a top-level module
lama_dir = os.path.dirname(__file__)
if lama_dir not in sys.path:
    sys.path.insert(0, lama_dir)

from .lama_inpainter import LaMaInpainter

__all__ = ['LaMaInpainter']
