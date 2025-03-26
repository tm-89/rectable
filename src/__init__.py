import torch
from icecream import install

torch.set_num_threads(1)
install()

from .data import *  # noqa
