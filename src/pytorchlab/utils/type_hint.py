from typing import Callable, Literal, Sequence

from torch import Tensor
from torch.nn import Module

Transforms = Callable[[Tensor], Tensor]
Stage = Literal["fit", "validate", "test", "predict"]

ImageShape = tuple[int, int] | tuple[int, int, int]
IntList = Sequence[int] | list[int]
FloatList = Sequence[float] | list[float]
ModuleList = Sequence[Module] | list[Module]
