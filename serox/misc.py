from __future__ import annotations
import copy
from typing import Protocol, Self, override, runtime_checkable

__all__ = [
    "Addable",
    "Clone",
    "Clone",
    "Divisible",
    "Divisible_",
    "IdentityFunction",
    "IndexType",
    "Indexable",
    "Multiplicable",
    "Multiplicable_",
    "SelfAddable",
    "SelfMultiplicable",
    "Sized",
    "SizedIndexable",
    "SizedSelfAddable",
]


type IndexType = int


class Clone(Protocol):
    """
    A common protocol for the ability to duplicate an object.
    """

    def clone(self) -> Self:
        """
        Duplicate `self`.
        """
        return copy.deepcopy(self)


class Dupe(Protocol):
    """
    A cheap version of :class:`~Clone`.
    """

    def dupe(self) -> Self:
        """
        Cheaply duplicate `self`.
        """
        return copy.copy(self)


@runtime_checkable
class Sized(Protocol):
    def __len__(self) -> int: ...


@runtime_checkable
class Indexable[_IndexItem](Protocol):
    def __getitem__(self, index: IndexType, /) -> _IndexItem: ...


@runtime_checkable
class SizedIndexable[_IndexItem](Sized, Indexable[_IndexItem], Protocol): ...


@runtime_checkable
class Multiplicable[Other](Protocol):
    def __mul__(self, other: Other) -> Self: ...


@runtime_checkable
class Multiplicable_[Other](Protocol):
    def __imul__(self, other: Other) -> Self: ...


@runtime_checkable
class Divisible[Other](Protocol):
    def __div__(self, other: Other) -> Self: ...


@runtime_checkable
class Divisible_[Other](Protocol):
    def __idiv__(self, other: Other) -> Self: ...


@runtime_checkable
class Addable[Other](Protocol):
    def __add__(self, other: Other, /) -> Self: ...


@runtime_checkable
class RAddable[Other](Protocol):
    def __radd__(self, other: Other, /) -> Self: ...


@runtime_checkable
class LAddable[Other](Protocol):
    def __ladd__(self, other: Other, /) -> Self: ...


@runtime_checkable
class SelfAddable(Protocol):
    def __add__(self, other: Self, /) -> Self: ...


@runtime_checkable
class SelfMultiplicable(Protocol):
    def __mul__(self, other: Self, /) -> Self: ...


@runtime_checkable
class SizedSelfAddable(
    Sized,
    SelfAddable,
    Protocol,
):
    @override
    def __len__(self) -> int: ...

    @override
    def __add__(self, other: Self) -> Self: ...


@runtime_checkable
class IdentityFunction(Protocol):
    def __call__[T](self, __x: T) -> T: ...
