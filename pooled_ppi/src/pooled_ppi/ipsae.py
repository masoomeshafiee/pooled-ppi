
import functools
import pandas as pd
from cached_path import cached_path
from .core import *

def read(path):
    frame = pd.read_csv(path, sep=r'\s+').query('Type == "max"')
    frame['name'] = frame['Model'].map(lambda s: os.path.basename(s).removesuffix('_model'))
    return frame
