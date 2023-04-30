import abc
import math
from dataclasses import field
from functools import partial
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    Iterable,
    ParamSpec,
    Self,
    TypeVar,
    cast,
)

from .serialize import StateDictSerializable

PSDefaultField = ParamSpec("PSDefaultField")
TDefaultField = TypeVar("TDefaultField")


def default_field(
    t: Callable[PSDefaultField, TDefaultField],
    *args: PSDefaultField.args,
    **kwargs: PSDefaultField.kwargs,
) -> TDefaultField:
    return field(
        default_factory=partial(t, *args, **kwargs),
    )


PSLoop = ParamSpec("PSLoop")
TLoop = TypeVar("TLoop")


def loop(
    fn: Callable[Concatenate[TLoop, PSLoop], Any],
    iter: Iterable[TLoop],
    *args: PSLoop.args,
    **kwargs: PSLoop.kwargs,
) -> None:
    ...
    for x in iter:
        fn(x, *args, **kwargs)


TAccumEl = TypeVar("TAccumEl")
TAccumAcc = TypeVar("TAccumAcc")
TAccumRed = TypeVar("TAccumRed")

AccumAccFn_t = Callable[[TAccumAcc, TAccumEl], TAccumAcc]
AccumRedFn_t = Callable[[TAccumAcc], TAccumRed]


class Accumulator(
    StateDictSerializable, Generic[TAccumAcc, TAccumEl, TAccumRed], abc.ABC,
):
    def __init__(
        self,
        acc: Callable[[], TAccumAcc],
    ) -> None:
        super().__init__()
        self._default_acc = acc
        self._acc: TAccumAcc = acc()
        self._count: int = 0
        self._total: int = 0

    def state_dict(self) -> dict[str, Any]:
        return {"acc": self._acc, "count": self._count, "total": self._total}

    def load_state_dict(self, state_dict: dict[str, Any]) -> Any:
        acc, count, total = map(state_dict.get, ("acc", "count", "total"))
        self._acc = acc or self._acc
        self._count = count or self._count
        self._total = total or self._total

    @property
    def acc(self) -> TAccumAcc:
        return self._acc

    @property
    def count(self) -> int:
        return self._count

    @property
    def total(self) -> int:
        return self._total

    @abc.abstractmethod
    def accumulate(self, acc: TAccumAcc, el: TAccumEl) -> TAccumAcc:
        ...

    @abc.abstractmethod
    def reduce(self, acc: TAccumAcc, count: int) -> TAccumRed:
        ...

    def flush(self) -> TAccumRed:
        out = self.reduce(self._acc, self._count)
        self._acc = self._default_acc()
        self._count = 0
        return out

    def add(self, el: TAccumEl, count: int = 1) -> Self:
        self._count += count
        self._total += count
        self._acc = self.accumulate(self._acc, el)
        return self

    def __iadd__(self, el: TAccumEl) -> Self:
        return self.add(el)


class RunningMean(Accumulator[float, float, float]):
    def __init__(self, acc: Callable[[], float] = lambda: 0.0) -> None:
        super().__init__(acc)

    def accumulate(self, acc: float, el: float) -> float:
        return acc + el

    def reduce(self, acc: float, count: int) -> float:
        if count == 0:
            return math.inf
        return acc / float(count)
