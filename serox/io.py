from typing import Protocol, runtime_checkable

__all__ = ["PathLike", "StrPath"]


@runtime_checkable
class PathLike[S: str | bytes](Protocol):
    def __fspath__(self) -> S: ...


type StrPath = str | PathLike[str]
