import abc
import os
from typing import Generic, TypeVar

from .serialize import SerializableData
from .state import ManagedState, SerializedStateData

TModelParam = TypeVar("TModelParam", bound=SerializableData)
TModelState = TypeVar("TModelState", bound=SerializableData)
TTrainParam = TypeVar("TTrainParam", bound=SerializableData)
TTrainState = TypeVar("TTrainState", bound=SerializableData)
ModelStateManager_t = ManagedState[TModelParam, TModelState]
TModelStateManager = TypeVar("TModelStateManager", bound=ModelStateManager_t)


class Trainer(
    Generic[TModelParam, TModelState, TTrainParam, TTrainState, TModelStateManager],
    abc.ABC,
):
    mstate_m: type[TModelStateManager]
    tstate_m: type[ManagedState[TTrainParam, TTrainState, TModelStateManager]]

    def __init__(
        self,
        mparam: TModelParam | SerializedStateData | str,
        tparam: TTrainParam | SerializedStateData | str,
    ) -> None:
        self._mstate = self.mstate_m(mparam)
        self._tstate = self.tstate_m(tparam, self._mstate)

    @property
    def mparam(self) -> TModelParam:
        return self._mstate.param
    
    @property
    def mstate(self) -> TModelState:
        return self._mstate.state
    
    @property
    def tparam(self) -> TTrainParam:
        return self._tstate.param
    
    @property
    def tstate(self) -> TTrainState:
        return self._tstate.state

    def save(self, dir: str) -> None:
        self._mstate.save(os.path.join(dir, "model.pt"))
        self._tstate.save(os.path.join(dir, "train.pt"))

    @abc.abstractmethod
    def train(self) -> None:
        ...
