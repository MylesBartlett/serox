from dataclasses import Field, dataclass, fields, is_dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from serox.result import Err, Ok, Result

__all__ = [
    "DataclassInstance",
    "shallow_asdict",
    "shallow_astuple",
]


@runtime_checkable
class DataclassInstance(Protocol):
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]


def shallow_astuple(dc: DataclassInstance, /) -> Result[tuple[Any, ...], TypeError]:
    """dataclasses.astuple() but without the deep-copying/recursion." """
    if not is_dataclass(dataclass):
        return Err(TypeError("shallow_astuple() should be called on dataclass instances"))
    return Ok(tuple(getattr(dc, field_.name) for field_ in fields(dc)))


def shallow_asdict(dc: DataclassInstance, /) -> Result[dict[str, Any], TypeError]:
    """dataclasses.asdict() but without the deep-copying/recursion." """
    if not is_dataclass(dc):
        return Err(TypeError("shallow_asdict() should be called on dataclass instances"))
    return Ok({field_.name: getattr(dc, field_.name) for field_ in fields(dc)})
