from typing import Protocol, Self

__all__ = [
    "From",
    "Into",
]


class From[T](Protocol):
    @classmethod
    def from_(cls, s: T, /) -> Self: ...


class Into[T](Protocol):
    def into(self) -> T: ...
