import abc

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


class LoggerAdapter(abc.ABC):
    def add_scalar(
        self,
        tag: Tag_t,
        scalar: Scalar_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        ...

    def add_histogram(
        self,
        tag: Tag_t,
        values: Hist_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        ...

    def add_image(
        self,
        tag: Tag_t,
        img: Img_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        ...

    def add_video(
        self,
        tag: Tag_t,
        video: Video_t,
        fps: FPS_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        ...

    def add_audio(
        self,
        tag: Tag_t,
        audio: Audio_t,
        sample_rate: SampleRate_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        ...

    def add_text(
        self,
        tag: Tag_t,
        text: Text_t,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        ...

    def add_embedding(
        self,
        tag: Tag_t,
        mat: EmbedMat_t,
        meta: EmbedMeta_t | None,
        img: Img_t | None,
        step: Step_t,
        wtime: Time_t,
    ) -> None:
        ...
