from typing import Protocol

__all__ = ["Default"]


class Default(Protocol):
    @classmethod
    def default[T](cls: type[T]) -> T: ...
