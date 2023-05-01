import abc
from typing import Any, Generic, NamedTuple, TypeVar, TypeVarTuple, overload

import torch

from .serialize import SerializableData

TParam = TypeVar("TParam", bound=SerializableData)
TState = TypeVar("TState", bound=SerializableData)
TConfArgs = TypeVarTuple("TConfArgs")


class SerializedStateData(NamedTuple):
    param: dict[str, Any]
    state: dict[str, Any]


class StateSerializer(Generic[TParam, TState, *TConfArgs], abc.ABC):
    param_t: type[TParam]
    state_t: type[TState]

    @classmethod
    def __init_subclass__(
        cls, param_t: type[TParam] | None, state_t: type[TState] | None
    ) -> None:
        if param_t is not None:
            cls.param_t: type[TParam] = param_t
        if state_t is not None:
            cls.state_t: type[TState] = state_t

    @classmethod
    @abc.abstractmethod
    def configure(cls, param: TParam, *args: *TConfArgs) -> TState:
        ...

    @classmethod
    def serialize(cls, param: TParam, state: TState) -> SerializedStateData:
        return SerializedStateData(
            cls.param_t.serialize(param), cls.state_t.serialize(state)
        )

    @classmethod
    def deserialize(
        cls, data: SerializedStateData, *conf_args: *TConfArgs
    ) -> tuple[TParam, TState]:
        param_d, state_d = data
        param = cls.param_t.deserialize(param_d)
        state = cls.configure(param, *conf_args)
        state = cls.state_t.deserialize(state_d, state)
        return param, state


class ManagedState(
    StateSerializer[TParam, TState, bool, *TConfArgs], abc.ABC, param_t=None, state_t=None
):
    @classmethod
    def __init_subclass__(cls, param_t: type[TParam], state_t: type[TState]) -> None:
        return super().__init_subclass__(param_t, state_t)

    @overload
    def __init__(self, p: TParam, *conf_args: *TConfArgs) -> None:
        ...

    @overload
    def __init__(self, p: str, *conf_args: *TConfArgs) -> None:
        ...

    @overload
    def __init__(self, p: SerializedStateData, *conf_args: *TConfArgs) -> None:
        ...

    def __init__(
        self,
        p: TParam | SerializedStateData | str,
        *conf_args: *TConfArgs,
    ) -> None:
        super().__init__()
        if isinstance(p, SerializableData):
            self.param: TParam = p
            self.state: TState = self.configure(p, True, *conf_args)
            return

        if isinstance(p, str):
            p = SerializedStateData(*torch.load(p))

        if not isinstance(p, tuple):
            raise TypeError(f"Invalid argument")

        self.param, self.state = self.deserialize(p, False, *conf_args)

    def serialize(self) -> SerializedStateData:
        return super().serialize(self.param, self.state)

    def save(self, path: str) -> None:
        torch.save(tuple(self.serialize()), path)
