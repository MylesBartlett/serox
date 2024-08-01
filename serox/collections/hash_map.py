from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generator, Hashable, Iterable, Literal, Sized, override

from serox.common import False_, True_
from serox.conftest import TESTING
from serox.convert import Into
from serox.default import Default
from serox.fmt import Debug
from serox.iter import Extend, FromIterator, IntoIterator, IntoParIterator, Iterator
from serox.misc import Clone
from serox.option import Null, Option, Some, is_null

__all__ = ["HashMap", "Entry"]

type Entry[K, V] = tuple[K, V]


class HashMap[K: Hashable, V](
    Default,
    Extend[Entry[K, V]],
    FromIterator[Entry[K, V]],
    IntoIterator[Entry[K, V]],
    IntoParIterator[Entry[K, V]],
    Into[dict[K, V]],
    Clone,
    Debug,
    Sized,
):
    def __init__(self, *args: Entry[K, V]) -> None:
        super().__init__()
        self.inner = {k: v for (k, v) in args}

    @override
    @classmethod
    def default(cls) -> HashMap[K, V]:
        return HashMap[K, V]()

    @classmethod
    def new(cls) -> HashMap[K, V]:
        return HashMap[K, V].default()

    @override
    def into(self) -> dict[K, V]:
        return self.inner

    @override
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({ repr(self.inner) })"

    def clear(self) -> None:
        self.inner.clear()

    def get(self, key: K, /) -> Option[V]:
        return Some[V].from_(self.inner.get(key))

    def entry(self, key: K, /) -> Option[Entry[K, V]]:
        return self.get(key).map(lambda v: (key, v))

    def insert(self, key: K, value: V) -> Option[V]:
        if is_null(opt := self.get(key)):
            self.inner[key] = value
        return opt

    def or_insert(self, default_key: K, default_value: V) -> Entry[K, V]:
        return (default_key, self.inner.setdefault(default_key, default_value))

    def contains_key(self, key: K) -> bool:
        return self.inner.__contains__(key)

    @override
    def __eq__(self, other: Any) -> bool:
        match other:
            case HashMap():
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

    @override
    def extend(self, iter: Iterable[Entry[K, V]], /) -> None:
        self.inner |= dict(iter)

    def remove(self, key: K, /) -> Option[V]:
        return Some[V].from_(self.inner.pop(key, None))

    def remove_entry(self, key: K, /) -> Option[Entry[K, V]]:
        return self.remove(key).map(lambda v: (key, v))

    @override
    @classmethod
    def from_iter(cls, iter: Iterable[Entry[K, V]], /) -> HashMap[K, V]:
        return HashMap(*iter)

    @override
    def iter(self) -> Entries[K, V, Literal[False]]:
        return Entries(self, par=False)

    def __iter__(self) -> Generator[Entry[K, V], None, None]:
        yield from self.iter()

    @override
    def par_iter(self) -> Entries[K, V, Literal[True]]:
        return Entries(self, par=True)

    def keys(self) -> Keys[K, Literal[False]]:
        return Keys[K, Literal[False]](self, par=False)

    def values(self) -> Values[V, Literal[False]]:
        return Values[V, Literal[False]](self, par=False)

    @override
    def clone(self) -> HashMap[K, V]:
        return HashMap[K, V](*self.inner.copy().items())


@dataclass(repr=True, init=False)
class Keys[K, P: (True_, False_)](Iterator[K, P]):
    def __init__(self, inner: HashMap[K, Any], /, par: P) -> None:
        super().__init__()
        self.iter = iter(inner.inner.keys())
        self.par = par

    @override
    def next(self) -> Option[K]:
        try:
            return Some(next(self.iter))
        except StopIteration:
            return Null()


@dataclass(repr=True, init=False)
class Values[V, P: (True_, False_)](Iterator[V, P]):
    def __init__(self, inner: HashMap[Any, V], /, par: P) -> None:
        super().__init__()
        self.iter = iter(inner.inner.values())
        self.par = par

    @override
    def next(self) -> Option[V]:
        try:
            return Some(self.iter.__next__())
        except StopIteration:
            return Null()


@dataclass(repr=True, init=False)
class Entries[K, V, P: (True_, False_)](Iterator[Entry[K, V], P]):
    def __init__(self, inner: HashMap[K, V], /, par: P) -> None:
        super().__init__()
        self.iter = iter(inner.inner.items())
        self.par = par

    @override
    def next(self) -> Option[Entry[K, V]]:
        try:
            return Some(next(self.iter))
        except StopIteration:
            return Null()


if TESTING:

    def test_hash_map():
        hm = HashMap(("foo", 0), ("bar", 1))
        _ = hm.iter().collect(HashMap[str, int])
        _ = hm.keys()
        _ = hm.values()
        print(hm)
