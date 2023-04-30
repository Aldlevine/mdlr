from typing import Protocol

import numpy as np
from torch import Tensor


class Stringable(Protocol):
    def __str__(self) -> str:
        ...


Audio_t = Tensor | np.ndarray
EmbedMat_t = Tensor | np.ndarray
EmbedMeta_t = list[Stringable]
FPS_t = int
Hist_t = Tensor | np.ndarray
Img_t = Tensor
SampleRate_t = int
Scalar_t = float | Tensor | np.ndarray
Step_t = int
Tag_t = str
Text_t = str
Time_t = float
Video_t = Tensor | np.ndarray
