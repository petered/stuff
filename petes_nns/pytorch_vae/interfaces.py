from typing import Callable
import torch
from torch.distributions import Distribution

IPositionToImageDecoder = Callable[[torch.Tensor], Distribution]
IImageToPositionEncoder = Callable[[torch.Tensor], Distribution]
