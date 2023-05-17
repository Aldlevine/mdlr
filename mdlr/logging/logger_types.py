from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from torch import Tensor


class Stringable(Protocol):
    def __str__(self) -> str:
        ...


Audio_t = Tensor
EmbedMat_t = Tensor
EmbedMeta_t = list[Stringable]
FPS_t = int
Hist_t = Tensor
Img_t = Tensor
SampleRate_t = int
Scalar_t = float | Tensor
Step_t = int
Tag_t = str
Text_t = str
Time_t = float
Video_t = Tensor
