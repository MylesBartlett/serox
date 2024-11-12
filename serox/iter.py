# pyright: reportImportCycles=none
from __future__ import annotations
from dataclasses import dataclass
import functools
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Protocol,
    Self,
    cast,
    final,
    override,
    runtime_checkable,
)
from typing import Iterator as NativeIterator

from joblib import (  # pyright: ignore[reportMissingTypeStubs]
    Parallel,
    delayed,  # pyright: ignore[reportUnknownVariableType]
)

from serox.cmp import Ord
from serox.common import False_, True_
from serox.conftest import TESTING
from serox.misc import SelfAddable, SelfMultiplicable

if TYPE_CHECKING:
    from serox.option import Null, Option, Some
    from serox.result import Result


__all__ = [
    "ArrayChunk",
    "Bridge",
    "Chain",
    "Chunk",
    "DoubleEndedIterator",
    "Extend",
    "Filter",
    "FilterMap",
    "IntoIterator",
    "Iterator",
    "Map",
    "Rev",
    "Take",
    "TakeWhile",
    "Zip",
]


type Fn1[T, U] = Callable[[T], U]


class FromIterator[A](Protocol):
    @classmethod
    def from_iter[P: (True_, False_)](cls, iter: Iterator[A, P], /) -> Self: ...


def _identity[T](x: T) -> T:
    """
    Identity function that returns the input value.
    """
    return x


@runtime_checkable
class Iterator[Item, Par: (True_, False_)](Protocol):
    par: Par

    def next(self) -> Option[Item]: ...

    def __next__(self) -> Item:
        from serox.option import Null, Some

        match self.next():
            case Some(x):
                return x
            case Null():
                raise StopIteration

    @final
    def _iter(self) -> Generator[Item, None, None]:
        from serox.option import Null, Some

        while True:
            match self.next():
                case Some(v):
                    yield v
                case Null():
                    break

    @final
    def __iter__(self) -> Generator[Item, None, None]:
        if self.par:
            yield from cast(
                Generator[Item, None, None],
                Parallel(n_jobs=-1, verbose=0)(delayed(_identity)(i) for i in self._iter()),
            )
        else:
            yield from self._iter()

    @final
    def zip[U](self, other: Iterator[U, Par]) -> Zip[Item, U, Par]:
        return Zip(a=self, b=other, par=self.par)

    @final
    def zip_longest[U](self, other: Iterator[U, Par]) -> ZipLongest[Item, U, Par]:
        return ZipLongest(a=self, b=other, par=self.par)

    @final
    def chain(self, other: Iterator[Item, Par]) -> Chain[Item, Par]:
        return Chain(a=self, b=other, par=self.par)

    @final
    def map[B](self, f: Fn1[Item, B], /) -> Map[Item, B, Par]:
        return Map(self, f, par=self.par)

    @final
    def fold[B](self, init: B, f: Callable[[B, Item], B], /) -> B:
        from .option import is_some

        accum = init
        while is_some(v := self.next()):
            accum = f(accum, v.unwrap())
        return accum

    def count(self) -> int:
        def increment(count: int, _: Item) -> int:
            return count + 1

        return self.fold(0, increment)

    def array_chunk(self, n: int, /) -> ArrayChunk[Item, Par]:
        return ArrayChunk(self, n, par=self.par)

    def filter(self, f: Fn1[Item, bool]) -> Filter[Item, Par]:
        return Filter(self, f, par=self.par)

    def filter_map[B](self, f: Fn1[Item, Option[B]]) -> FilterMap[Item, B, Par]:
        return FilterMap(self, f, par=self.par)

    @final
    def collect[B: "FromIterator[Any]"](self, collection: type[B], /) -> B:
        return collection.from_iter(self)

    def nth(self, n: int) -> Option[Item]:
        from serox.option import Null, Some

        while n > 0:
            match self.next():
                case Some(_):
                    n -= 1
                case Null():
                    return Null()
        return self.next()

    def advance_by(self, n: int, /) -> Result[None, int]:
        from .option import Null, Some
        from .result import Err, Ok

        while n > 0:
            match self.next():
                case Some(_):
                    n -= 1
                case Null():
                    return Err(n)
        return Ok(None)

    def for_each(self, f: Fn1[Item, None]) -> None:
        def call(_: None, item: Item, /) -> None:
            return f(item)

        self.fold(None, call)

    def take(self, n: int, /) -> Take[Item, Par]:
        return Take(self, n, par=self.par)

    def take_while(self, f: Fn1[Item, bool], /) -> TakeWhile[Item, Par]:
        return TakeWhile(self, f, par=self.par)

    def sum[U: SelfAddable](self: Iterator[U, Par]) -> U:
        return functools.reduce(operator.add, self)

    def product[U: SelfMultiplicable](self: Iterator[U, Par]) -> U:
        return functools.reduce(operator.mul, self)

    def all(self, f: Fn1[Item, bool], /) -> bool:
        return all(f(item) for item in self)

    def any(self, f: Fn1[Item, bool], /) -> bool:
        return any(f(item) for item in self)

    def max[U: Ord](self: Iterator[U, Par]) -> U:
        return max(self)

    def min[U: Ord](self: Iterator[U, Par]) -> U:
        return min(self)

    def par_bridge(self) -> Iterator[Item, True_]:
        object.__setattr__(self, "par", True)
        return cast(Iterator[Item, True_], self)


class Chunk[Item](list[Item], FromIterator[Item]):
    @override
    @classmethod
    def from_iter(cls, iter: Iterable[Item], /) -> Chunk[Item]:
        return Chunk(iter)

    def len(self) -> int:
        return self.__len__()

    def is_empty(self) -> bool:
        return self.len() == 0


@dataclass
class ArrayChunk[Item, P: (True_, False_)](Iterator[Chunk[Item], P]):
    iter: Iterator[Item, P]
    n: int
    par: P

    @override
    def next(self) -> Option[Chunk[Item]]:
        from .option import Null, Some

        match self.iter.take(self.n).collect(Chunk[Item]):
            case []:
                return Null()
            case chunk:
                return Some(chunk)


class IntoIterator[T](Protocol):
    def iter(self) -> Iterator[T, False_]: ...


class IntoParIterator[T](Protocol):
    def par_iter(self) -> Iterator[T, True_]: ...


class DoubleEndedIterator[Item, P: (True_, False_)](Iterator[Item, P], Protocol):
    def next_back(self) -> Option[Item]: ...

    def rev(self) -> Rev[Item, P]:
        return Rev(self, par=self.par)


@dataclass(repr=True)
class Filter[Item, P: (True_, False_)](Iterator[Item, P]):
    iter: Iterator[Item, P]
    f: Fn1[Item, bool]
    par: P

    @override
    def next(self) -> Option[Item]:
        from serox.option import Null, Some

        match self.iter.next():
            case Some(v) if self.f(v):
                return Some(v)
            case Some(v):
                return self.next()
            case Null():
                return Null()


@dataclass(repr=True)
class FilterMap[Item, B, P: (True_, False_)](Iterator[B, P]):
    iter: Iterator[Item, P]
    f: Fn1[Item, Option[B]]
    par: P

    @override
    def next(self) -> Option[B]:
        from serox.option import Null, Some

        match self.iter.next():
            case Some(x):
                match y := self.f(x):
                    case Some(_):
                        return y
                    case Null():
                        return self.next()
            case Null():
                return Null()


@dataclass(repr=True)
class Map[Item, B, P: (True_, False_)](Iterator[B, P]):
    iter: Iterator[Item, P]
    f: Fn1[Item, B]
    par: P

    @override
    def next(self) -> Option[B]:
        return self.iter.next().map(self.f)


@dataclass(repr=True)
class Take[Item, P: (True_, False_)](Iterator[Item, P]):
    iter: Iterator[Item, P]
    _n: int
    par: P

    @override
    def next(self) -> Option[Item]:
        if self._n != 0:
            self._n -= 1
            return self.iter.next()
        else:
            from .option import Null

            return Null()

    @override
    def nth(self, n: int) -> Option[Item]:
        if self._n > n:
            self._n -= n + 1
            return self.iter.nth(n)
        else:
            if self._n > 0:
                _ = self.iter.nth(self.n - 1)
                self.n = 0
            return Null()


@dataclass(repr=True)
class TakeWhile[Item, P: (True_, False_)](Iterator[Item, P]):
    iter: Iterator[Item, P]
    predicate: Fn1[Item, bool]
    par: P
    flag: bool = False

    @override
    def next(self) -> Option[Item]:
        from serox.option import Null, Some

        if self.flag:
            return Null()
        else:
            match self.iter.next():
                case Null():
                    return Null()
                case Some(x) if self.predicate(x):
                    return Some(x)
                case _:
                    self.flag = True
                    return Null()


@dataclass(repr=True)
class Zip[A, B, P: (True_, False_)](Iterator[tuple[A, B], P]):
    a: Iterator[A, P]
    b: Iterator[B, P]
    par: P

    @override
    def next(self) -> Option[tuple[A, B]]:
        from serox.option import Null, Some

        match self.a.next():
            case Null():
                return Null()
            case Some(val_a):
                match self.b.next():
                    case Null():
                        return Null()
                    case Some(val_b):
                        return Some((val_a, val_b))


# Parametrising the first generic of `Iterator` as `Any` to avoid a circular import.
@dataclass(repr=True)
class ZipLongest[A, B, P: (True_, False_)](Iterator[Any, P]):
    a: Iterator[A, P]
    b: Iterator[B, P]
    par: P

    @override
    def next(self) -> Option[tuple[A, B] | tuple[Null[A], B] | tuple[A, Null[B]]]:
        from serox.option import Null, Some

        opt_a, opt_b = self.a.next(), self.b.next()
        match (opt_a, opt_b):
            case (Some(val_a), Some(val_b)):
                return Some((val_a, val_b))
            case (Null(), Some(val_b)):
                return Some((Null[A](), val_b))
            case (Some(val_a), Null()):
                return Some((val_a, Null[B]()))
            case _:
                return Null()


@dataclass(repr=True)
class Chain[A, P: (True_, False_)](Iterator[A, P]):
    a: Iterator[A, P]
    b: Iterator[A, P]
    par: P

    @override
    def next(self) -> Option[A]:
        from serox.option import Null, Some

        match item := self.a.next():
            case Null():
                return self.b.next()
            case Some(_):
                return item


@dataclass(repr=True)
class Rev[Item, P: (True_, False_)](Iterator[Item, P]):
    iter: DoubleEndedIterator[Item, P]
    par: P

    @override
    def next(self) -> Option[Item]:
        return self.iter.next_back()


class Extend[Item](Protocol):
    def extend(self, iter: Iterable[Item], /) -> None: ...

    def extend_one(self, item: Item) -> None:
        self.extend(Some(item))


@dataclass(repr=True, frozen=True, kw_only=True)
class Bridge[Item, Par: (True_, False_)](Iterator[Item, Par]):
    """
    A bridge between native Python iterators and `serox` ones.
    Can be parallel (`par = True`) or non-parallel (`par = False`).
    """

    iter: NativeIterator[Item]
    """The native Python iterator being bridged."""
    par: Par
    """Whether to parallelise the iterator."""

    @classmethod
    def new[Item2, Par2: (True_, False_)](
        cls, iter: NativeIterator[Item2], /, par: Par2 = False
    ) -> Bridge[Item2, Par2]:
        return Bridge(iter=iter, par=par)

    @classmethod
    def par_new[Item2](cls, iter: NativeIterator[Item2], /) -> Bridge[Item2, True_]:
        return Bridge(iter=iter, par=True)

    @override
    def next(self) -> Option[Item]:
        from .option import Null, Some

        try:
            return Some(self.iter.__next__())
        except StopIteration:
            return Null()


if TESTING:

    def test_par_invariance():
        from .collections import HashMap
        from .vec import Vec

        values = Vec(*range(4))
        keys = ["foo", "bar", "baz"]
        bridge = Bridge.new(iter(keys), par=False)
        bridge = Bridge(iter=iter(keys), par=False)
        mapped = values.iter().map(lambda x: x**2)
        _ = bridge.zip(mapped).collect(HashMap[str, int])

        bridge = Bridge.new(iter(keys), par=True)
        # shouldn't be able to combine parallel iterators with non-parallel ones
        # for consistent typing
        _ = bridge.zip(values.iter())  # pyright: ignore[reportArgumentType]

    def test_max():
        """Test the max method for iterators."""
        from .vec import Vec

        vec = Vec(1, 2, 3, 4, 5)
        assert vec.iter().max() == 5

    def test_iterator_methods():
        from .option import Null, Some
        from .result import Err, Ok
        from .vec import Vec

        vec = Vec(1, 2, 3, 4, 5)
        iter = vec.iter()

        # Test next
        assert iter.next() == Some(1)
        assert iter.next() == Some(2)

        # Test nth
        assert iter.nth(2) == Some(5)
        assert iter.nth(1) == Null()

        # Test advance_by
        iter = vec.iter()
        assert iter.advance_by(3) == Ok(None)
        assert iter.advance_by(3) == Err(1)

        # Test for_each
        result: list[int] = []
        iter = vec.iter()
        iter.for_each(lambda x: result.append(x))
        assert result == [1, 2, 3, 4, 5]

        # Test take
        iter = vec.iter().take(3)
        assert list(iter) == [1, 2, 3]

        # Test take_while
        iter = vec.iter().take_while(lambda x: x < 4)
        assert list(iter) == [1, 2, 3]

        # Test sum
        assert vec.iter().sum() == 15

        # Test product
        assert vec.iter().product() == 120

        # Test all
        assert vec.iter().all(lambda x: x > 0)
        assert not vec.iter().all(lambda x: x < 3)

        # Test any
        assert vec.iter().any(lambda x: x == 3)
        assert not vec.iter().any(lambda x: x == 6)

        # Test max
        assert vec.iter().max() == 5

        # Test min
        assert vec.iter().min() == 1

    def test_combinational_methods():
        from .option import Null, Some
        from .vec import Vec

        vec1 = Vec(1, 2, 3)
        vec2 = Vec(4, 5, 6)
        iter1 = vec1.iter()
        iter2 = vec2.iter()

        # Test zip
        zipped = iter1.zip(iter2)
        assert list(zipped) == [(1, 4), (2, 5), (3, 6)]

        # Test zip_longest
        iter1 = vec1.iter()
        iter2 = Vec(4, 5).iter()
        zipped_longest = iter1.zip_longest(iter2)
        assert list(zipped_longest) == [(1, 4), (2, 5), (3, Null())]

        # Test chain
        iter1 = vec1.iter()
        iter2 = vec2.iter()
        chained = iter1.chain(iter2)
        assert list(chained) == [1, 2, 3, 4, 5, 6]

        # Test map
        iter = vec1.iter().map(lambda x: x * 2)
        assert list(iter) == [2, 4, 6]

        # Test filter
        iter = vec1.iter().filter(lambda x: x % 2 == 1)
        assert list(iter) == [1, 3]

        # Test filter_map
        iter = vec1.iter().filter_map(lambda x: Some(x * 2) if x % 2 == 0 else Null())
        assert list(iter) == [4]

    def test_parallel_iterators():
        from .option import Null, Some
        from .result import Err, Ok
        from .vec import Vec

        vec = Vec(1, 2, 3, 4, 5)
        iter = vec.par_iter()

        # Test next
        assert iter.next() == Some(1)
        assert iter.next() == Some(2)

        # Test nth
        assert iter.nth(2) == Some(5)
        assert iter.nth(1) == Null()

        # Test advance_by
        iter = vec.par_iter()
        assert iter.advance_by(3) == Ok(None)
        assert iter.advance_by(3) == Err(1)

        # Test for_each
        result: list[int] = []
        iter = vec.par_iter()
        iter.for_each(lambda x: result.append(x))
        assert result == [1, 2, 3, 4, 5]

        # Test take
        iter = vec.par_iter().take(3)
        assert list(iter) == [1, 2, 3]

        # Test take_while
        iter = vec.par_iter().take_while(lambda x: x < 4)
        assert list(iter) == [1, 2, 3]

        # Test sum
        assert vec.par_iter().sum() == 15

        # Test product
        assert vec.par_iter().product() == 120

        # Test all
        assert vec.par_iter().all(lambda x: x > 0)
        assert not vec.par_iter().all(lambda x: x < 3)

        # Test any
        assert vec.par_iter().any(lambda x: x == 3)
        assert not vec.par_iter().any(lambda x: x == 6)

        # Test max
        assert vec.par_iter().max() == 5

        # Test min
        assert vec.par_iter().min() == 1
