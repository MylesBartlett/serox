from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generator, Hashable, Iterable, Literal, Self, Sized, override

from serox.convert import Into
from serox.default import Default
from serox.iter import Extend, FromIterator, IntoIterator, IntoParIterator, Iterator
from serox.misc import Clone
from serox.option import Null, Option, Some

__all__ = [
    "HashSet",
    "Iter",
]


class HashSet[T: Hashable](
    Default,
    Extend[T],
    FromIterator[T],
    IntoIterator[T],
    IntoParIterator[T],
    Into[set[T]],
    Clone,
    Sized,
):
    def __init__(self, *args: T) -> None:
        super().__init__()
        self.inner = set(args)

    @override
    @classmethod
    def default(cls) -> HashSet[T]:
        return HashSet[T]()

    @classmethod
    def new(cls) -> HashSet[T]:
        return HashSet[T].default()

    @override
    def into(self) -> set[T]:
        return self.inner

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({ repr(self.inner) })"

    def clear(self) -> None:
        self.inner.clear()

    def difference(self, other: Self) -> HashSet[T]:
        return HashSet[T](*(self.inner.difference(other.inner)))

    def __sub__(self, other: Self) -> HashSet[T]:
        return self.difference(other)

    def symmetric_difference(self, other: Self) -> HashSet[T]:
        return HashSet[T](*(self.inner.symmetric_difference(other.inner)))

    def union(self, other: Self) -> HashSet[T]:
        return HashSet[T](*(self.inner.union(other.inner)))

    def __or__(self, other: Self) -> HashSet[T]:
        return self.union(other)

    def intersection(self, other: Self) -> HashSet[T]:
        return HashSet[T](*(self.inner.intersection(other.inner)))

    def __and__(self, other: Self) -> HashSet[T]:
        return self.intersection(other)

    def __xor__(self, other: Self) -> HashSet[T]:
        return self.symmetric_difference(other)

    def contains(self, value: T) -> bool:
        return self.inner.__contains__(value)

    def is_disjoint(self, other: Self) -> bool:
        return self.inner.isdisjoint(other.inner)

    def is_subset(self, other: Self) -> bool:
        return self.inner.issubset(other.inner)

    def is_superset(self, other: Self) -> bool:
        return self.inner.issuperset(other.inner)

    def insert(self, value: T) -> bool:
        if preexisting := self.contains(value):
            self.inner.add(value)
        return preexisting

    @override
    def __eq__(self, other: Any) -> bool:
        match other:
            case HashSet():
                return self.inner == other.inner  # pyright: ignore[reportUnknownMemberType]
            case _:
                return False

    def len(self) -> int:
        return self.inner.__len__()

    @override
    def __len__(self) -> int:
        return self.len()

    def is_empty(self) -> bool:
        return self.len() == 0

    def take(self, value: T) -> Option[T]:
        if self.contains(value):
            self.remove(value)
            return Some(value)
        return Null()

    @override
    def extend(self, iter: Iterable[T], /) -> None:
        self.inner.update(iter)

    def remove(self, value: T, /) -> None:
        self.inner.remove(value)

    @override
    @classmethod
    def from_iter(cls, iter: Iterable[T], /) -> HashSet[T]:
        return HashSet(*iter)

    @override
    def iter(self) -> Iter[T, Literal[False]]:
        return Iter(self, par=False)

    def __iter__(self) -> Generator[T, None, None]:
        yield from self.iter()

    @override
    def par_iter(self) -> Iter[T, Literal[True]]:
        return Iter(self, par=True)

    @override
    def clone(self) -> HashSet[T]:
        return HashSet(*self.inner.copy())


@dataclass(repr=True, init=False)
class Iter[Item, P: bool](Iterator[Item, P]):
    def __init__(self, inner: HashSet[Item], /, par: P) -> None:
        super().__init__()
        self.iter = iter(inner.inner)
        self.par = par

    @override
    def next(self) -> Option[Item]:
        try:
            return Some(next(self.iter))
        except StopIteration:
            return Null()
