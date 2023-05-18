import time
from typing import Any, Callable, Concatenate, NotRequired, ParamSpec, Unpack, cast, TYPE_CHECKING

from ..serialize import StateDictSerializable
from .logger_adapter import LoggerAdapter
if TYPE_CHECKING:
    from .logger_types import (
        Audio_t,
        EmbedMat_t,
        EmbedMeta_t,
        FPS_t,
        Hist_t,
        Img_t,
        SampleRate_t,
        Scalar_t,
        Step_t,
        Tag_t,
        Text_t,
        Time_t,
        Video_t,
    )

PSApply = ParamSpec("PSApply")

SharedParams_t = dict[
    {
        "step": "NotRequired[Step_t | None]",
        "inc": "NotRequired[Step_t | None]",
        "wtime": "NotRequired[Time_t | None]",
    }
]


class Logger(StateDictSerializable):
    def __init__(
        self,
        *adapters: LoggerAdapter,
    ) -> None:
        self._adapters = list(adapters)
        self._tag_steps: dict[Tag_t, Step_t] = {}

    def add_adapters(self, *adapter: LoggerAdapter) -> None:
        self._adapters += list(adapter)

    def state_dict(self) -> dict[str, Any]:
        return self._tag_steps

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        self._tag_steps = state_dict

    def _apply(
        self,
        fn: Callable[Concatenate[Any, PSApply], None],
        *args: PSApply.args,
        **kwargs: PSApply.kwargs,
    ) -> None:
        for adpt in self._adapters:
            adpt_fn = getattr(adpt, fn.__name__)
            adpt_fn(*args, **kwargs)

    def _step_time(self, tag: "Tag_t", params: SharedParams_t) -> tuple["Step_t", "Time_t"]:
        inc = cast(Step_t, params.get("inc", 1))
        step = params.get("step", None)
        wtime = cast(Time_t, params.get("wtime", time.time()))

        if step == None:
            step = self._tag_steps.get(tag, 0) + inc
        self._tag_steps[tag] = step

        return step, wtime

    def add_scalar(
        self,
        tag: "Tag_t",
        scalar: "Scalar_t",
        **args: Unpack[SharedParams_t],
    ) -> None:
        step, wtime = self._step_time(tag, args)
        self._apply(LoggerAdapter.add_scalar, tag, scalar, step, wtime)

    def add_histogram(
        self,
        tag: "Tag_t",
        values: "Hist_t",
        **args: Unpack[SharedParams_t],
    ) -> None:
        step, wtime = self._step_time(tag, args)
        self._apply(LoggerAdapter.add_histogram, tag, values, step, wtime)

    def add_image(
        self,
        tag: "Tag_t",
        img: "Img_t",
        **args: Unpack[SharedParams_t],
    ) -> None:
        """_summary_

        Args:
            tag (Tag_t): _description_
            img (Img_t): _description_
        Shape:
            img: (C, H, W) | (N, C, H, W) | (Y, X, C, H, W)
        """

        import torch

        step, wtime = self._step_time(tag, args)

        dim = img.dim()
        assert dim in (
            3,
            4,
            5,
        ), "Image tensor must be (C, H, W) | (N, C, H, W) | (Y, X, C, H, W)"

        channels = img.shape[-3]
        assert channels in (1, 3), "Image tensor must have 1 or 3 channels"

        if dim == 4:
            img = torch.cat(img.unbind(0), -1)
        elif dim == 5:
            img = torch.cat(torch.cat(img.unbind(0), -2).unbind(0), -1)

        self._apply(LoggerAdapter.add_image, tag, img, step, wtime)

    def add_video(
        self,
        tag: "Tag_t",
        video: "Video_t",
        fps: "FPS_t" = 4,
        **args: Unpack[SharedParams_t],
    ) -> None:
        step, wtime = self._step_time(tag, args)
        self._apply(LoggerAdapter.add_video, tag, video, fps, step, wtime)

    def add_audio(
        self,
        tag: "Tag_t",
        audio: "Audio_t",
        sample_rate: "SampleRate_t" = 44_100,
        **args: Unpack[SharedParams_t],
    ) -> None:
        step, wtime = self._step_time(tag, args)
        self._apply(LoggerAdapter.add_audio, tag, audio, sample_rate, step, wtime)

    def add_text(
        self,
        tag: "Tag_t",
        text: "Text_t",
        **args: Unpack[SharedParams_t],
    ) -> None:
        step, wtime = self._step_time(tag, args)
        self._apply(LoggerAdapter.add_text, tag, text, step, wtime)

    def add_embedding(
        self,
        tag: "Tag_t",
        mat: "EmbedMat_t",
        meta: "EmbedMeta_t | None" = None,
        img: "Img_t | None" = None,
        **args: Unpack[SharedParams_t],
    ) -> None:
        step, wtime = self._step_time(tag, args)
        self._apply(LoggerAdapter.add_embedding, tag, mat, meta, img, step, wtime)
