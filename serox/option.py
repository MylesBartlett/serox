# pyright: reportImportCycles=none
from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import (
    Callable,
    Generator,
    Hashable,
    NoReturn,
    Protocol,
    TypeGuard,
    final,
    overload,
    override,
)

from serox.common import False_, True_
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
    def iter(self: Option[T]) -> Iter[T, False_]:
        """
        Returns an iterator over the possibly contained value.

        Examples
        ========
        .. code-block:: python
            x = Some(4)
            assert x.iter().next() == 4

            x: Option[int] = Null()
            assert x.iter().next() == Null()
        """
        return Iter(self, par=False)

    def __next__(self: Option[T]) -> Option[T]:
        return self.iter().next()

    def __iter__(self: Option[T]) -> Generator[T, None, None]:
        yield from Iter(self, par=False)

    def or_(self: Option[T], optb: Option[T], /) -> Option[T]:
        """
        Returns the option if it contains a value, otherwise returns `optb`.

        Arguments passed to or are eagerly evaluated; if you are passing the result of a function
        call, it is recommended to use :meth:`or_else`, which is lazily evaluated.
        """
        return self.__or__(optb)

    def __or__(self: Option[T], optb: Option[T], /) -> Option[T]:
        match x := self:
            case Some(_):
                return x
            case Null():
                return optb

    def or_else(self: Option[T], f: Fn0[Option[T]], /) -> Option[T]:
        """
        Returns the option if it contains a value, otherwise calls `f` and returns the result.
        """
        match x := self:
            case Some(_):
                return x
            case Null():
                return f()

    # Property emulative of Rust's `?` operator when
    # used in conjunction with [`serox::question_mark::qmark`].
    @property
    def q(self: Option[T]) -> T:
        """
        '?' operator for early exiting an `Option` returning function.

        Examples
        ========
        .. code-block:: python

            @qmark
            def some_function(value: Option[str]) -> Option[str]:
                return Some(value.q + "_suffix")
        """
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
        """
        Returns `True` if the option is a `Some` value.
        """
        return isinstance(self, Some)

    def is_null(self: Option[T]) -> bool:
        """
        Returns `True` if the option is a `Null` value.
        """
        return not self.is_some()

    @overload
    def unwrap(self: Some[T]) -> T: ...
    @overload
    def unwrap(self: Null[T]) -> NoReturn: ...
    def unwrap(self: Option[T]) -> T | NoReturn:
        """
        Returns the contained `Some` value, consuming the `self` value.

        Because this function may panic, its use is generally discouraged.
        Instead, prefer to use pattern matching and handle the `Null` case explicitly, or call
        :meth:`unwrap_or`, :meth:`unwrap_or_else`, or :meth:`unwrap_or_default`.

        raises :class:`UnwrapFailed`: if the `self` is value equals `Null`
        """
        match self:
            case Some(x):
                return x
            case Null():
                raise UnwrapFailed()

    def unwrap_or(self: Option[T], default: T, /) -> T:
        """
        Returns the contained `Some` value or a provided `default`.

        Arguments passed to :meth:`unwrap_or` are eagerly evaluated; if you are passing the result
        of a function call, it is recommended to use :meth:`unwrap_or_else`, which is lazily
        evaluated.

        Examples
        ========
        .. code-block:: python

            assert Some("car").unwrap_or("bike") == "car"
            assert Null[str].unwrap_or("bike") == "bike"
        """
        match self:
            case Some(x):
                return x
            case Null():
                return default

    def unwrap_or_else(self: Option[T], f: Callable[[], T], /) -> T:
        """
        Returns the contained `Some` value or computes it from a closure.

        Examples
        ========
        .. code-block:: python

            k = 10
            assert Some(4).unwrap_or_else(lambda: 2 * k) == 4
            assert Null[int]().unwrap_or_else(lambda: 2 * k) == 20
        """
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
        """
        Maps an `Option[T]` to `Option[U]` by applying a function to a contained value (if `Some`)
        or returns `Null` (if `Null`).

        Examples
        ========
        .. code-block:: python

                maybe_some_string = Some("Hello, World!")
                maybe_some_len = maybe_some_string.map(lambda s: len(s))
                assert maybe_some_len == Some(13)

                x: Option[str] = Null()
                assert x.map(lambda s: len(s)) == Null()
        """
        match self:
            case Some(t):
                return Some(f(t))
            case Null():
                return Null[U]()

    def map_or[U](self: Option[T], default: U, f: Fn1[T, U], /) -> U:
        """
        Returns the provided default result (if `Null`), or applies a function to the contained
        value (if `Null`).

        Arguments passed to map_or are eagerly evaluated; if you are passing the result of a
        function call, it is recommended to use map_or_else, which is lazily evaluated.

        Examples
        ========
        .. code-block:: python

            x = Some("foo")
            assert x.map_or(42, lambda v: len(v)) == 3

            x: Option[str] = Null()
            assert x.map_or(42, lambda v: len(v)) == 42
        """
        match self:
            case Some(x):
                return f(x)
            case Null():
                return default

    def map_or_else[U](self: Option[T], default: Fn0[U], f: Fn1[T, U], /) -> U:
        """
        Computes a default function result (if `Null`), or applies a different function to the
        contained value (if `Some`).

        Examples
        ========
        .. code-block:: python

            k = 21
            x = Some("foo")
            assert x.map_or_else(lambda: 2 * k, lambda v: len(v)) == 3

            x: Option[str] = Null()
            assert x.map_or_else(lambda: 2 * k, lambda v: len(v)) == 42
        """
        match self:
            case Some(x):
                return f(x)
            case Null():
                return default()

    @overload
    def and_[U](self: Null[T], optb: Option[U], /) -> Null[U]: ...
    @overload
    def and_[U](self: Some[T], optb: Some[U], /) -> Some[U]: ...
    @overload
    def and_[U](self: Some[T], optb: Null[U], /) -> Null[U]: ...
    def and_[U](self: Option[T], optb: Option[U], /) -> Option[U]:
        """
        Returns `Null` if the option is `Null`, otherwise returns `optb`.

        Arguments passed to and are eagerly evaluated; if you are passing the result of a function
        call, it is recommended to use :meth:`and_then`, which is lazily evaluated.

        Examples
        ========

        .. code-block:: python

            x = Some(2)
            y: Option[str] = Null()
            assert x.and_(y) == Null()

            x: Option[int] = Null()
            y = Some("foo")
            assert x.and_(y) == Null()

            x = Some(2)
            y = Some("foo")
            assert x.and_(y) == Some("foo")

            x: Option[int] = Null()
            y: Option[str] = Null()
            assert x.and_(y) == Null()
        """
        match self:
            case Null():
                return Null[U]()
            case Some(_):
                return optb

    @overload
    def __and__[U](self: Null[T], optb: Option[U], /) -> Null[U]: ...
    @overload
    def __and__[U](self: Some[T], optb: Some[U], /) -> Some[U]: ...
    @overload
    def __and__[U](self: Some[T], optb: Null[U], /) -> Null[U]: ...
    def __and__[U](self: Option[T], optb: Option[U], /) -> Option[U]:
        return self.and_(optb)

    @overload
    def and_then[U](self: Null[T], f: Fn1[T, U], /) -> Null[U]: ...
    @overload
    def and_then[U](self: Some[T], f: Fn1[T, U], /) -> U: ...
    def and_then[U](self: Option[T], f: Fn1[T, U], /) -> U | Null[U]:
        """
        Returns `Null` if the option is `Null` otherwise calls `f` with the wrapped value and returns
        the result.

        Some languages call this operation 'flatmap'.

        Examples
        ========
        .. code-block:: python

            def sqrt_then_to_string(x: int) -> Option[str]:
                if x < 0:
                    return Null()
                return Some(str(x**0.5))


            assert Some(2).and_then(sq_then_to_string) == Some(str(4))
            assert Some(-1).and_then(sq_then_to_string) == Null()
            assert Null[str]().and_then(sq_then_to_string) == Null()

        Often used to chain fallible operations that may return `Null`.

        .. code-block:: python

            arr_2d = Vec(Vec("A0", "A1"), Vec("B0", "B1"))
            item_0_0 = arr_2d.first().and_then(lambda row: row.first())
            assert item_0_0 == Some("A1")

            item_2_0 = arr_2d.get(2).and_then(lambda row: row.get(0))
            assert item_2_0 == Null()
        """
        match self:
            case Some(x):
                return f(x)
            case Null():
                return Null[U]()

    def ok_or[E: Exception](self: Option[T], err: E, /) -> Result[T, E]:
        """
        Transforms the `Option[T]` into a `Result[T, E]`, mapping `Some(v)` to `Ok(v)` and `Null`
        to `Err(err)`.

        Arguments passed to :meth:`ok_or` are eagerly evaluated; if you are passing the result of a
        function call, it is recommended to use :meth:`ok_or_else`, which is lazily evaluated.

        Examples
        ========
        .. code-block:: python

            x = Some("foo")
            assert x.ok_or(0) == Ok("foo")

            x: Option[str] = Null()
            assert x.ok_or(0) == Err(0)
        """
        match self:
            case Some(x):
                return Ok(x)
            case Null():
                return Err(err)

    def ok_or_else[E: Exception](self: Option[T], err: Fn0[E], /) -> Result[T, E]:
        """
        Transforms the `Option[T]` into a `Result[T, E]`, mapping `Some(v)` to `Ok(v)` and `Null` to
        `Err(err())`.

        Examples
        =======
        .. code-block:: python

            x = Some("foo")
            assert x.ok_or_else(lambda: 0) == Ok("foo")

            x: Option[str] = Null()
            assert x.ok_or_else(lambda: 0) == Err(0)
        """
        match self:
            case Some(x):
                return Ok(x)
            case Null():
                return Err(err())

    def filter(self: Option[T], f: Fn1[T, bool]) -> Option[T]:
        """
        Returns `Null` if the option is `Null`, otherwise calls predicate with the wrapped value
        and returns:

        - `Some(t)` if predicate returns `True` (where `t` is the wrapped value), and
        - `None` if predicate returns `False`.

        This function works similar to :meth:`Iterator.filter()`.
        You can imagine the `Option[T]` being an iterator over one or zero elements.
        `filter()` lets you decide which elements to keep.
        """

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
        """
        Returns `Null`.
        """
        return Null()

    def zip[U](self: Option[T], other: Option[U]) -> Option[tuple[T, U]]:
        """
        Zips `self` with another `Option`.

        If `self` is `Some(s)` and `other` is `Some(o)`, this method returns `Some((s, o))`.
        Otherwise, `Null` is returned.

        Examples
        ========
        .. code-block:: python

            x = Some(1)
            y = Some("hi")
            z = Null[int]

            assert x.zip(y) == Some((1, "hi"))
            assert x.zip(z) == Null()
        """
        match (self, other):
            case (Some(a), Some(b)):
                return Some((a, b))
            case _:
                return Null()

    def transpose[E](self: Option[Result[T, E]]) -> Result[Option[T], E]:
        """
        Transposes an `Option` of a `Result` into a `Result` of an `Option`.

        `Null` will be mapped to `Ok(Null())`. `Some(Ok(_))` and `Some(Err(_))` will be mapped to
        `Ok(Some(_))` and `Err(_)`.

        Examples
        ========
        .. code-block:: python

            x = Result[Option[int], ValueError] = Ok(Some(5))
            y = Option[Result[int], ValueError] = Some(Ok(5))
            assert x == y.transpose()
        """
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
    """
    No value.
    """

    __slots__ = ()

    # Emulate [`NoneType`].
    def __bool__(self) -> False_: ...


@final
@dataclass(
    eq=True,
    repr=True,
    slots=True,
    frozen=True,
)
class Some[T](_Option[T]):
    """
    Some value of type `T`.
    """

    __match_args__ = ("value",)
    value: T


type Option[T] = Some[T] | Null[T]
"""
Type `Option` represents an optional value: every `Option` is either `Some` and contains a value,
or `Null`, and does not. `Option` types are very common in Rust code, as they have a number of uses:

- Initial values
- Return values for functions that are not defined over their entire input range (partial functions)
- Return value for otherwise reporting simple errors, where `Null` is returned on error
- Optional class attributes
- Optional function arguments

Options are commonly paired with pattern matching to query the presence of a value and take action,
always accounting for the `Null` case.

.. code-block:: python

    def divide(numerator: float, denominator: float) -> Option[float]:
        if denominator == 0.0:
            return Null()
        else:
            return Some(numerator / denominator)

    # The return value of the function is an option
    result = divide(2.0, 3.0)

    # Pattern match to retrieve the value
    match result:
        #  The division was valid
        case Some(x):
            print(f"Result {x}")
        # The division was invalid
        case Null():
            print("Cannot divide by 0")
"""


def is_some[T](x: Option[T], /) -> TypeGuard[Some[T]]:
    """
    Returns `True` if the option is a `Some` value.

    Since `TypeGuard` does not allow for type-narrowing of the `self` parameter,
    this external function is needed to achieve that goal; :meth:`~Option.is_some`
    on its own does not suffice.
    """
    return x.is_some()


def is_null[T](x: Option[T], /) -> TypeGuard[Null[T]]:
    """
    Returns `True` if the option is a `Null` value.

    Since `TypeGuard` does not allow for type-narrowing of the `self` parameter,
    this external function is needed to achieve that goal; :meth:`~Option.is_null`
    on its own does not suffice.
    """
    return x.is_null()


@dataclass(
    repr=True,
    slots=True,
)
class Iter[Item, P: (True_, False_)](DoubleEndedIterator[Item, P]):
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
