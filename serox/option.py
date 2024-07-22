# pyright: reportImportCycles=none
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import (
    Callable,
    Generator,
    Hashable,
    Literal,
    NoReturn,
    Protocol,
    TypeGuard,
    final,
    overload,
    override,
)

from serox.convert import From, Into
from serox.default import Default
from serox.iter import DoubleEndedIterator, IntoIterator
from serox.misc import Clone
from serox.result import Err, Ok, Result

from .conftest import TESTING

__all__ = [
    "Null",
    "Option",
    "Some",
    "is_null",
    "is_some",
]


@final
class UnwrapFailed(Exception):
    @override
    def __init__(self) -> None:
        super().__init__("Called `Option.unwrap()` on a Null value")


@final
class NullShortCircuit(Exception): ...


type Fn0[T] = Callable[[], T]
type Fn1[T, U] = Callable[[T], U]


class _Option[T](
    Clone,
    Default,
    Hashable,
    From[T | None],
    Into[T | None],
    IntoIterator[T],
    Protocol,
):
    @override
    def iter(self: Option[T]) -> Iter[T, Literal[False]]:
        return Iter(self, par=False)

    def __next__(self: Option[T]) -> Option[T]:
        return self.iter().next()

    def __iter__(self: Option[T]) -> Generator[T, None, None]:
        yield from Iter(self, par=False)

    def or_(self: Option[T], optb: Option[T], /) -> Option[T]:
        return self.__or__(optb)

    def __or__(self: Option[T], optb: Option[T], /) -> Option[T]:
        match x := self:
            case Some(_):
                return x
            case Null():
                return optb

    def or_else(self: Option[T], f: Fn0[Option[T]], /) -> Option[T]:
        match x := self:
            case Some(_):
                return x
            case Null():
                return f()

    # Property emulative of Rust's ? operator when
    # used in conjunction with [`as_option`].
    @property
    def q(self: Option[T]) -> T:
        match self:
            case Some(x):
                return x
            case Null():
                raise NullShortCircuit

    def __invert__(self: Option[T]) -> T:
        return self.q

    @overload
    @classmethod
    def from_(cls: type[Some[T]], val: T, /) -> Some[T]: ...
    @overload
    @classmethod
    def from_(cls: type[Null[T]], val: None, /) -> Null[T]: ...
    @overload
    @classmethod
    def from_(cls: type[Some[T]], val: None, /) -> Null[T]: ...
    @overload
    @classmethod
    def from_(cls: type[Null[T]], val: T, /) -> Some[T]: ...
    @overload
    @classmethod
    def from_(cls: type[Option[T]], val: T | None, /) -> Option[T]: ...
    @override
    @classmethod
    def from_(cls: type[Option[T]], val: T | None, /) -> Option[T]:
        match val:
            case None:
                return Null[T]()
            case _:
                return Some(val)

    @override
    def into(self: Option[T]) -> T | None:
        match self:
            case Some(x):
                return x
            case Null():
                return None

    def is_some(self: Option[T]) -> bool:
        return isinstance(self, Some)

    def is_null(self: Option[T]) -> bool:
        return not self.is_some()

    @overload
    def unwrap(self: Some[T]) -> T: ...
    @overload
    def unwrap(self: Null[T]) -> NoReturn: ...
    def unwrap(self: Option[T]) -> T | NoReturn:
        match self:
            case Some(x):
                return x
            case Null():
                raise UnwrapFailed()

    def unwrap_or(self: Option[T], default: T, /) -> T:
        match self:
            case Some(x):
                return x
            case Null():
                return default

    def unwrap_or_else(self: Option[T], f: Callable[[], T], /) -> T:
        match self:
            case Some(x):
                return x
            case Null():
                return f()

    @overload
    def map[U](self: Null[T], f: Fn1[T, U], /) -> Null[U]: ...
    @overload
    def map[U](self: Some[T], f: Fn1[T, U], /) -> Some[U]: ...
    def map[U](self: Option[T], f: Fn1[T, U], /) -> Option[U]:
        match self:
            case Some(t):
                return Some(f(t))
            case Null():
                return Null[U]()

    def map_or[U](self: Option[T], default: U, f: Fn1[T, U], /) -> U:
        match self:
            case Some(x):
                return f(x)
            case Null():
                return default

    def map_or_else[U](self: Option[T], default: Fn0[U], f: Fn1[T, U], /) -> U:
        match self:
            case Some(x):
                return f(x)
            case Null():
                return default()

    @overload
    def and_then[U](self: Null[T], f: Fn1[T, U], /) -> Null[U]: ...
    @overload
    def and_then[U](self: Some[T], f: Fn1[T, U], /) -> U: ...
    def and_then[U](self: Option[T], f: Fn1[T, U], /) -> U | Null[U]:
        match self:
            case Some(x):
                return f(x)
            case Null():
                return Null[U]()

    def ok_or[E: Exception](self: Option[T], err: E, /) -> Result[T, E]:
        match self:
            case Some(x):
                return Ok(x)
            case Null():
                return Err(err)

    def ok_or_else[E: Exception](self: Option[T], err: Fn0[E], /) -> Result[T, E]:
        match self:
            case Some(x):
                return Ok(x)
            case Null():
                return Err(err())

    def filter(self: Option[T], f: Fn1[T, bool]) -> Option[T]:
        match self:
            case Some(x) if f(x):
                return Some(x)
            case _:
                return Null()

    @overload
    def clone(self: Null[T]) -> Null[T]: ...
    @overload
    def clone(self: Some[T]) -> Some[T]: ...
    @override
    def clone(self: Option[T]) -> Option[T]:
        match self:
            case Some(v):
                return Some(copy.deepcopy(v))
            case Null():
                return Null[T]()

    @override
    @classmethod
    def default(cls: type[Option[T]]) -> Option[T]:
        return Null()

    def zip[U](self: Option[T], other: Option[U]) -> Option[tuple[T, U]]:
        """
        Zips `self` with another `Option`.

        If `self` is `Some(s)` and `other` is `Some(o)`, this method returns `Some((s, o))`.
        Otherwise, `Null` is returned.
        """
        match (self, other):
            case (Some(a), Some(b)):
                return Some((a, b))
            case _:
                return Null()

    def transpose[E](self: Option[Result[T, E]]) -> Result[Option[T], E]:
        from .result import Err, Ok

        match self:
            case Some(Ok(x)):
                return Ok(Some(x))
            case Some(Err(e)):
                return Err(e)
            case _:
                return Ok(Null())


@final
@dataclass(
    eq=True,
    frozen=True,
    repr=True,
)
class Null[T](_Option[T]):
    __slots__ = ()

    # Emulate [`NoneType`].
    def __bool__(self) -> Literal[False]: ...


@final
@dataclass(
    eq=True,
    repr=True,
    slots=True,
    frozen=True,
)
class Some[T](_Option[T]):
    __match_args__ = ("value",)
    value: T


type Option[T] = Some[T] | Null[T]


# Applying [`TypeGuard`] to a `self` parameter is not permitted so
# we have to resort to defining external `is_some` and `is_null` functions
# in order to generate boolean checkers capable of type narrowing.
def is_some[T](x: Option[T], /) -> TypeGuard[Some[T]]:
    return x.is_some()


def is_null[T](x: Option[T], /) -> TypeGuard[Null[T]]:
    return x.is_null()


@dataclass(
    repr=True,
    slots=True,
)
class Iter[Item, P: bool](DoubleEndedIterator[Item, P]):
    item: Option[Item]
    par: P

    @override
    def next(self) -> Option[Item]:
        match self.item:
            case Some(x):
                self.item = Null()
                return Some(x)
            case Null():
                return Null()

    @override
    def next_back(self) -> Option[Item]:
        return self.item


if TESTING:
    from typing import cast

    import pytest

    @pytest.fixture()
    def foo_optional() -> str | None:
        return "foo"

    def test_from_into(foo_optional: str | None):
        opt = Some[str].from_(foo_optional)
        assert opt.unwrap() == "foo"
        opt_ls = cast(list[str] | None, ["bar", "baz"])
        opt = Some[list[str]].from_(opt_ls)
        assert opt.unwrap() == ["bar", "baz"]
        opt = Some[str].from_(None)
        with pytest.raises(UnwrapFailed):
            _ = opt.unwrap()

        deopted = opt.into()
        assert deopted is None

    @pytest.fixture()
    def foo_option() -> Option[str]:
        return Some("foo")

    def test_matching(foo_option: Option[str]):
        match foo_option:
            case Some(v):
                assert v == "foo"
            case Null():
                raise ValueError
        option2 = cast(Option[str], Null[str]())
        match option2:
            case Some(v):
                raise ValueError
            case v:
                ...

    def test_map(foo_option: Option[str]):
        res = foo_option.map(lambda x: x + "bar")
        assert res.unwrap() == "foobar"
        foo_option = Null[str]()
        res = foo_option.unwrap_or_else(lambda: "baz")
        assert res == "baz"

    def test_iter(foo_option: Option[str]):
        val = next(foo_option)
        assert val == Some("foo")
        val = list(iter(foo_option))
        assert len(val) == 1

        ls: list[str] = []
        ls.extend(foo_option)
        assert len(ls) == 1
        assert ls[0] == "foo"

        foo_option = Null[str]()
        val = list(iter(foo_option))
        assert len(val) == 0
