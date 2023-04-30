from typing import Literal

from torch.utils.tensorboard.writer import SummaryWriter

from ..logger_adapter import LoggerAdapter
from ..logger_types import (
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

TagList_t = list[Tag_t]
ChartType_t = Literal["Margin", "Multiline"]
Chart_t = tuple[ChartType_t, TagList_t]
Category_t = dict[str, Chart_t]
Layout_t = dict[str, Category_t]


class TensorboardAdapter(LoggerAdapter):
    def __init__(self, logdir: str, layout: Layout_t | None = None) -> None:
        super().__init__()

        self._writer = SummaryWriter(logdir)

        if layout is not None:
            self._writer.add_custom_scalars(layout)

    def add_scalar(
        self,
        tag: Tag_t,
        scalar: Scalar_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        self._writer.add_scalar(tag, scalar, step, wtime, new_style=True)

    # TODO: add additional args?
    def add_histogram(
        self,
        tag: Tag_t,
        values: Hist_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        self._writer.add_histogram(tag, values, step, walltime=wtime)

    def add_image(
        self,
        tag: Tag_t,
        img: Img_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        self._writer.add_image(tag, img, step, wtime)

    def add_video(
        self,
        tag: Tag_t,
        video: Video_t,
        fps: FPS_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        self._writer.add_video(tag, video, step, fps, wtime)

    def add_audio(
        self,
        tag: Tag_t,
        audio: Audio_t,
        sample_rate: SampleRate_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        self._writer.add_audio(tag, audio, step, sample_rate, wtime)

    def add_text(
        self,
        tag: Tag_t,
        text: Text_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        self._writer.add_text(tag, text, step, wtime)

    def add_embedding(
        self,
        tag: Tag_t,
        mat: EmbedMat_t,
        meta: EmbedMeta_t | None,
        img: Img_t | None,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        self._writer.add_embedding(mat, meta, img, step, tag)
