from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Generator,
    Hashable,
    NoReturn,
    Protocol,
    TypeGuard,
    cast,
    final,
    overload,
    override,
)

from serox.conftest import TESTING
from serox.convert import Into
from serox.misc import Clone

from .fmt import Debug

if TYPE_CHECKING:
    from serox.option import Option

__all__ = [
    "Ok",
    "Err",
    "Result",
    "is_ok",
    "is_err",
]


@final
class UnwrapFailed(Exception):
    @override
    def __init__(self, msg: str, error: Debug, /) -> None:
        super().__init__(f"{msg}: {error.__repr__}")


type Fn0[T] = Callable[[], T]
type Fn1[T, U] = Callable[[T], U]


class _Result[T, E](
    Clone,
    Hashable,
    Protocol,
):
    def __iter__(self: Result[T, E]) -> Generator[T, None, None]:
        match self:
            case Ok(x):
                yield x
            case Err():
                ...

    def __next__(self: Result[T, E]) -> T:
        match self:
            case Ok(x):
                return x
            case Err(_):
                raise StopIteration

    # Property emulative of Rust's `?` operator when
    # used in conjunction with [`serox::question_mark::qmark`].
    @property
    def q(self: Result[T, E]) -> T:
        match self:
            case Ok(t):
                return t
            case Err(err):
                raise ErrShortCircuit(err)

    def __invert__(self: Result[T, E]) -> T:
        return self.q

    def unwrap(self: Result[T, E]) -> T:
        match self:
            case Ok(x):
                return x
            case Err(err):
                raise UnwrapFailed("called `Result.unwrap()` on an `Err` value", err)

    def unwrap_or(self: Result[T, E], default: T) -> T:
        match self:
            case Ok(x):
                return x
            case Err(_):
                return default

    def unwrap_or_else(self: Result[T, E], f: Fn0[T]) -> T:
        match self:
            case Ok(x):
                return x
            case Err(_):
                return f()

    @overload
    def expect(self: Ok[T, E], msg: str) -> T: ...
    @overload
    def expect(self: Err[T, E], msg: str) -> NoReturn: ...
    def expect(self: Result[T, E], msg: str) -> T | NoReturn:
        match self:
            case Ok(t):
                return t
            case Err(e):
                raise UnwrapFailed(msg, e)

    @overload
    def expect_err(self: Ok[T, E], msg: str) -> NoReturn: ...
    @overload
    def expect_err(self: Err[T, E], msg: str) -> E: ...
    def expect_err(self: Result[T, E], msg: str) -> E | NoReturn:
        match self:
            case Ok(t):
                raise UnwrapFailed(msg, t)
            case Err(e):
                return e

    @overload
    def map[U](self: Err[T, E], f: Fn1[T, U], /) -> Err[U, E]: ...
    @overload
    def map[U](self: Ok[T, E], f: Fn1[T, U], /) -> Ok[U, E]: ...
    def map[U](self: Result[T, E], f: Fn1[T, U], /) -> Result[U, E]:
        match self:
            case Ok(t):
                return Ok(f(t))
            case Err(e):
                return Err[U, E](e)

    @overload
    def map_err[F](self: Err[T, E], f: Fn1[T, F], /) -> Err[T, F]: ...
    @overload
    def map_err[F](self: Ok[T, E], f: Fn1[T, F], /) -> Ok[T, F]: ...
    def map_err[F](self: Result[T, E], f: Fn1[E, F], /) -> Result[T, F]:
        match self:
            case Ok(t):
                return Ok[T, F](t)
            case Err(e):
                return Err[T, F](f(e))

    def map_or[U](self: Result[T, E], default: U, f: Fn1[T, U], /) -> U:
        match self:
            case Ok(x):
                return f(x)
            case Err(_):
                return default

    def map_or_else[U](self: Result[T, E], default: Fn0[U], f: Fn1[T, U], /) -> U:
        match self:
            case Ok(x):
                return f(x)
            case Err(_):
                return default()

    def is_ok(self: Result[T, E]) -> bool:
        return isinstance(self, Ok)

    def is_err(self: Result[T, E]) -> bool:
        return not self.is_ok()

    def ok(self: Result[T, E]) -> Option[T]:
        from .option import Null, Some

        match self:
            case Ok(x):
                return Some(x)
            case Err(_):
                return Null[T]()

    def err(self: Result[T, E]) -> Option[E]:
        from .option import Null, Some

        match self:
            case Ok(_):
                return Null()
            case Err(x):
                return Some(x)

    @overload
    def and_then[U](self: Err[T, E], f: Fn1[T, U], /) -> Err[U, E]: ...
    @overload
    def and_then[U](self: Ok[T, E], f: Fn1[T, U], /) -> Result[U, E]: ...
    def and_then[U](self: Result[T, E], op: Callable[[T], Result[U, E]], /) -> Result[U, E]:
        match self:
            case Ok(t):
                return op(t)
            case Err(e):
                return Err[U, E](e)

    def or_(self: Result[T, E], res: Result[T, E], /) -> Result[T, E]:
        return self.__or__(res)

    def __or__(self: Result[T, E], res: Result[T, E], /) -> Result[T, E]:
        match self:
            case Ok(t):
                return Ok(t)
            case Err():
                return res

    def or_else[F](self: Result[T, E], op: Fn1[E, Result[T, F]], /) -> Result[T, F]:
        match self:
            case Ok(t):
                return Ok(t)
            case Err(e):
                return op(e)

    @override
    def clone(self: Result[T, E]) -> Result[T, E]:
        match self:
            case Ok(x):
                return Ok(copy.deepcopy(x))
            case Err(x):
                return Err(copy.deepcopy(x))

    def transpose(self: Result[Option[T], E]) -> Option[Result[T, E]]:
        from .option import Null, Some

        match self:
            case Ok(Some(x)):
                return Some(Ok(x))
            case Err(e):
                return Some(Err(e))
            case _:  # Ok(Null())
                return Null()


@final
@dataclass(
    eq=True,
    repr=True,
    frozen=True,
)
class Ok[T, E](_Result[T, E]):
    __match_args__ = ("value",)
    __slots__ = ("value",)

    value: T


@final
@dataclass(
    eq=True,
    frozen=True,
    repr=True,
)
class Err[T, E](_Result[T, E]):
    __match_args__ = ("value",)
    __slots__ = ("value",)

    value: E


type Result[T, E] = Ok[T, E] | Err[T, E]


# Applying [`TypeGuard`] to a `self` parameter is not permitted so
# we have to resort to defining external `is_ok` and `is_err` functions
# in order to generate boolean checkers capable of type narrowing.
def is_ok[T, E](x: Result[T, E], /) -> TypeGuard[Ok[T, E]]:
    return x.is_ok()


def is_err[T, E](x: Result[T, E], /) -> TypeGuard[Err[T, E]]:
    return x.is_err()


class ErrShortCircuit[T, E](Into[Err[T, E]], Exception):
    __match_args__ = ("err",)

    def __init__(self, err: E) -> None:
        super().__init__()
        self.err = err

    @override
    def into(self) -> Err[T, E]:
        return Err[T, E](self.err)


if TESTING:
    from typing import cast

    import pytest

    @pytest.fixture()
    def foo_res() -> Ok[str, Exception]:
        return Ok("foo")

    def test_matching(foo_res: Result[str, Exception]):
        match foo_res:
            case Ok(v):
                assert v == "foo"
            case Err(err):
                raise err
        option2 = cast(Result[str, Exception], Err("bar"))
        match option2:
            case Ok(v):
                raise ValueError
            case v:
                ...

    def test_map(foo_res: Result[str, Exception]):
        res = foo_res.map(lambda x: x + "bar")
        assert res.unwrap() == "foobar"
        foo_res = Err[str, Exception](Exception("Error"))
        res = foo_res.unwrap_or_else(lambda: "baz")
        assert res == "baz"

    def test_expect(foo_res: Result[str, Exception]):
        ok = foo_res.expect("Expected value 'foo'")
        assert ok == "foo"
        with pytest.raises(UnwrapFailed):
            _ = foo_res.expect_err("Expected error")
        err = Err[str, str]("Some error")
        with pytest.raises(UnwrapFailed):
            err.expect("Expected value 'foo'")

    def test_iter(foo_res: Result[str, Exception]):
        # Ok[T, _] should yield a single item of type T
        val = next(foo_res)
        assert val == "foo"
        val = list(iter(foo_res))
        assert len(val) == 1

        ls: list[str] = []
        ls.extend(foo_res)
        assert len(ls) == 1
        assert ls[0] == "foo"

        # Err[T, _] should yield nothing
        err = Err[str, str]("Error")
        val = list(iter(err))
        assert len(val) == 0
