from __future__ import annotations
import copy
from dataclasses import dataclass, field
from types import EllipsisType
from typing import Sized, override

from serox.conftest import TESTING
from serox.fmt import Debug
from serox.iter import DoubleEndedIterator
from serox.misc import Clone
from serox.option import Null, Option, Some

from .common import False_, True_

__all__ = ["Range"]

type Idx = int


@dataclass(
    eq=True,
    frozen=True,
    init=False,
    repr=False,
)
class Range[Par: (True_, False_)](
    DoubleEndedIterator[Idx, Par],
    Clone,
    Debug,
    Sized,
):
    __match_args__ = ("start", "end")

    start: Idx = field(init=False)
    end: Idx = field(init=False)
    par: Par = field(init=False)
    _ptr: Idx = field(init=False)

    def __init__(self, start: Idx | EllipsisType, end: Idx, *, par: Par = False) -> None:
        start = 0 if start is ... else start
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "par", par)
        object.__setattr__(self, "_ptr", start)
        super().__init__()

    @classmethod
    def new[Par2: (True_, False_)](
        cls, start: Idx | EllipsisType, end: Idx, *, par: Par2 = False
    ) -> Range[Par2]:
        return Range(start, end, par=par)

    @override
    def next(self) -> Option[Idx]:
        if self._ptr < self.end:
            item = Some(copy.copy(self._ptr))
            object.__setattr__(self, "_ptr", self._ptr + 1)
            return item
        return Null()

    @override
    def next_back(self) -> Option[Idx]:
        if self._ptr < self.end:
            item = Some(self.end - 1 - self._ptr)
            object.__setattr__(self, "_ptr", self._ptr + 1)
            return item
        return Null()

    @classmethod
    def from_sized[Par2: (True_, False_)](
        cls, sized: Sized, /, start: int = 0, *, par: Par2 = False
    ) -> Range[Par2]:
        return Range[Par2](start, len(sized), par=par)

    def contains(self, item: Idx, /) -> bool:
        return self.start <= item < self.end

    def is_empty(self) -> bool:
        return not (self.start < self.end)

    @override
    def clone(self) -> Range[Par]:
        return Range(start=self.start, end=self.end, par=self.par)

    def len(self) -> int:
        return max(self.end - self.start - 1, 0)

    @override
    def __len__(self) -> int:
        return self.len()

    @override
    def __repr__(self) -> str:
        return f"{self.start}..{self.end}"


if TESTING:
    from serox.vec import Vec

    def test_range():
        obj = Range(..., 10)
        _ = obj.par_bridge().map(lambda x: x**2).collect(Vec[int])
        for _ in obj:
            ...
