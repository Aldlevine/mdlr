from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

import src.state as state
from src.serialize import SerializableData
from src.utils import default_field


@dataclass(kw_only=True)
class ConvSpec(SerializableData):
    kernel_size: int = 3
    stride: int
    padding: int = 1


@dataclass(kw_only=True)
class ModelParam(SerializableData):
    in_channels: int = 3
    "Number of channels in input image"
    aux_width: int = 0
    "Number of classes (unconditional when 0)"
    downsamples: tuple[int, ...] = (16, 32, 64)
    "Number of channels at each downsample layer"
    residual_width: int = 64
    "Number of channels in residual layers"
    residual_depth: int = 4
    "Number of residual layers for both encoder and decoder"
    bottleneck: int = 4
    "Number of channels at bottleneck layer"
    flat_spec: ConvSpec = default_field(ConvSpec, kernel_size=3, stride=1, padding=1)
    "Spec for non scaling conv2d"
    pool_spec: ConvSpec = default_field(ConvSpec, kernel_size=3, stride=2, padding=1)
    "Spec for pooling conv2d"
    unpool_spec: ConvSpec = default_field(ConvSpec, kernel_size=2, stride=2, padding=0)
    "Spec for unpooling conv2d"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    "torch.device"


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._module(x)


class Conv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spec: ConvSpec,
        aux_width: int = 0,
        residual: bool = False,
        ctype: type[nn.Conv2d] | type[nn.ConvTranspose2d] = nn.Conv2d,
    ) -> None:
        super().__init__()

        self.residual = residual
        self.conv = ctype(in_channels, out_channels, **asdict(spec))
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

        if aux_width > 0:
            self.aux = nn.Conv2d(aux_width, in_channels, 1)
        else:
            self.aux = None

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        if self.aux is not None:
            assert aux is not None
            aux = self.aux(aux)
            assert aux is not None
            aux = aux.repeat(1, 1, x.shape[-2], x.shape[-1])
        else:
            aux = torch.zeros_like(x)

        skip = x
        x = self.conv(x + aux)
        x = self.norm(x)
        if self.residual:
            x = x + skip
        x = self.act(x)
        return x

    def __call__(
        self, x: torch.Tensor, aux: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward(x, aux)


class Model(nn.Module):
    def __init__(
        self,
        param: ModelParam,
    ) -> None:
        super().__init__()

        channels = [param.downsamples[0], *param.downsamples]
        rchannels = list(reversed(channels))

        self.in_conv = Conv(param.in_channels, channels[0], param.flat_spec)

        self.down_conv = nn.ModuleList(
            [
                Conv(ch0, ch1, param.pool_spec, aux_width=param.aux_width)
                for ch0, ch1 in zip(channels[:-1], channels[1:])
            ]
        )

        self.down_res = nn.ModuleList(
            [
                Conv(
                    param.residual_width,
                    param.residual_width,
                    param.flat_spec,
                    aux_width=param.aux_width,
                    residual=True,
                )
                for _ in range(param.residual_depth)
            ]
        )

        self.bottleneck_down = Conv(
            param.residual_width, param.bottleneck, param.flat_spec
        )
        self.bottleneck_up = Conv(
            param.bottleneck, param.residual_width, param.flat_spec
        )

        self.up_res = nn.ModuleList(
            [
                Conv(
                    param.residual_width,
                    param.residual_width,
                    param.flat_spec,
                    residual=True,
                )
                for _ in range(param.residual_depth)
            ]
        )

        self.up_conv = nn.ModuleList(
            [
                Conv(ch0, ch1, param.unpool_spec, ctype=nn.ConvTranspose2d, aux_width=param.aux_width)
                for ch0, ch1 in zip(rchannels[:-1], rchannels[1:])
            ]
        )

        self.out_conv = nn.Conv2d(
            param.downsamples[0], param.in_channels, **asdict(param.flat_spec)
        )

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        x = self.in_conv(x)
        for l in self.down_conv:
            x = l(x, aux)
        for l in self.down_res:
            x = l(x, aux)
        x = self.bottleneck_down(x)
        x = self.bottleneck_up(x)
        for l in self.up_res:
            x = l(x, aux)
        for l in self.up_conv:
            x = l(x, aux)
        x = self.out_conv(x)
        x = torch.tanh(x)
        return x

    def __call__(
        self, x: torch.Tensor, aux: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.forward(x, aux)


@dataclass(kw_only=True)
class ModelState(SerializableData):
    model: Model


class ModelStateManager(
    state.ManagedState[ModelParam, ModelState],
    param_t=ModelParam,
    state_t=ModelState,
):
    @classmethod
    def configure(cls, param: ModelParam) -> ModelState:
        return ModelState(model=Model(param).to(param.device))
