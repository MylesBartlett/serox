from typing import Any, Protocol, Self, override, runtime_checkable

__all__ = ["PartialOrd", "Ord"]


@runtime_checkable
class PartialOrd(Protocol):
    def __lt__(self, other: Self, /) -> bool: ...


@runtime_checkable
class Ord(Protocol):
    def __lt__(self, other: Self, /) -> bool: ...
    @override
    def __eq__(self, other: Any, /) -> bool: ...
