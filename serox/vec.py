from __future__ import annotations  # noqa: I001
from dataclasses import dataclass
from random import Random as Rng
from .common import True_, False_
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    Iterable,
    Self,
    overload,
    override,
)

from serox.convert import Into
from serox.fmt import Debug
from serox.question_mark import qmark
from serox.result import ErrShortCircuit, Ok, Result, Err
from .iter import (
    DoubleEndedIterator,
    FromIterator,
    IntoIterator,
    IntoParIterator,
    Iterator,
    Extend,
)
from .cmp import PartialOrd
from .default import Default


from serox.conftest import TESTING

from serox.misc import Clone, IndexType, SizedIndexable
from serox.option import Null, Option, Some

if TYPE_CHECKING:
    from serox import Range

__all__ = [
    "Vec",
]

type Fn0[T] = Callable[[], T]
type Fn1[T, U] = Callable[[T], U]


@dataclass(repr=True, frozen=True, kw_only=True)
class Iter[Item, P: (True_, False_)](DoubleEndedIterator[Item, P]):
    data: SizedIndexable[Item]
    par: P
    end_or_len: int
    ptr: int

    """
    Wrapper around a native Python :class:`list` to endow it with `Vec`-like functionality.
    """

    @classmethod
    def new[Item2, Par2: (True_, False_)](
        cls, data: SizedIndexable[Item2], /, par: Par2 = False
    ) -> Iter[Item2, Par2]:
        return Iter(
            data=data,
            par=par,
            ptr=0,
            end_or_len=len(data),
        )

    def _next(self, back: bool) -> Option[Item]:
        if self.ptr < self.end_or_len:
            ptr = -self.ptr if back else self.ptr
            item = Some(self.data[ptr])
            object.__setattr__(self, "ptr", self.ptr + 1)
            return item
        return Null()

    @override
    def next(self) -> Option[Item]:
        return self._next(back=False)

    @override
    def next_back(self) -> Option[Item]:
        return self._next(back=True)


@dataclass(repr=True, frozen=True, kw_only=True)
class SampledIndexable[Item](SizedIndexable[Item]):
    data: SizedIndexable[Item]
    indices: list[IndexType]

    @classmethod
    def new[Item2](
        cls,
        data: SizedIndexable[Item2],
        indices: list[IndexType],
    ) -> SampledIndexable[Item2]:
        return SampledIndexable(data=data, indices=indices)

    @override
    def __getitem__(self, index: IndexType, /) -> Item:
        return self.data[self.indices[index]]

    @override
    def __len__(self) -> int:
        return len(self.indices)


class Vec[T](
    IntoIterator[T],
    IntoParIterator[T],
    SizedIndexable[T],
    FromIterator[T],
    Default,
    Debug,
    Clone,
    Extend[T],
    Into[list[T]],
):
    """
    Wrapper around a native Python :class:`list` to endow it with `Vec`-like functionality.
    """

    def __init__(self, *args: T) -> None:
        super().__init__()
        self.inner = list(args)

    @override
    @classmethod
    def default(cls: type[Vec[T]]) -> Vec[T]:
        """
        Returns a new, empty `Vec` as the default.

        :returns: A new, empty `Vec`.
        """
        return Vec[T]()

    @classmethod
    def new(cls) -> Vec[T]:
        """
        Returns a new, empty `Vec`.

        :returns: A new, empty `Vec`.
        """
        return Vec[T].default()

    @override
    def into(self) -> list[T]:
        return self.inner

    @override
    def iter(self) -> DoubleEndedIterator[T, False_]:
        """
        Returns an iterator over the underlying list.

        The iterator yields all items from start to end.

        :returns: A double-ended iterator over the underlying list.
        """
        return Iter.new(self, par=False)

    @override
    def par_iter(self) -> DoubleEndedIterator[T, True_]:
        """
        Returns a parallel iterator over the underlying list.

        The iterator yields all items from start to end.

        :returns: A parallel double-ended iterator over the underlying list.
        """
        return Iter.new(self, par=True)

    def __iter__(self) -> Generator[T, None, None]:
        """
        Returns a generator over the elements of the `Vec`.

        :yield: Elements of the `Vec` in order.
        """
        yield from self.iter()

    @override
    def __eq__(self, other: Any) -> bool:
        match other:
            case Vec():
                return self.inner == other.inner  # pyright: ignore[reportUnknownMemberType]
            case _:
                return False

    @override
    def __repr__(self) -> str:
        return f"Vec({self.inner.__repr__()[1:-1]})"

    @override
    def __getitem__(self, index: int, /) -> T:
        return self.inner[index]

    @overload
    def get(self, index: Range[Any], /) -> Option[Vec[T]]: ...
    @overload
    def get(self, index: int, /) -> Option[T]: ...
    def get(self, index: int | Range[Any], /) -> Option[T] | Option[Vec[T]]:
        """
        Returns an element or sub-vector depending on the type of index.

        - If given a position, returns the element at that position or `Null` if out of bounds.
        - If given a range, returns the sub-vector corresponding to that range, or `Null` if out of
        bounds.
        """
        from serox import Range

        match index:
            case Range():
                if index.contains(self.len()):
                    return Some(Vec(*self.inner[index.start : index.end]))
                return Null()
            case int():
                if index >= self.len():
                    return Null()
                return Some(self.inner[index])

    @override
    def __len__(self) -> int:
        return len(self.inner)

    def len(self) -> int:
        """
        Returns the number of elements in the vector, also referred to as its 'length'.
        """
        return self.__len__()

    def is_empty(self) -> bool:
        """
        :returns: `True` if the vector contains no elements.
        """
        return self.len() == 0

    def split_off(self, at: int, /) -> Vec[T]:
        """
        Splits the vector into two at the given index, retaining the head :math:`[0, at)` and
        returning the tail :math:`[at. len)`.

        :param at: The index to split the vector at.
        :returns: A newly-allocated vector containing the elements :math:`[at, len)` of the tail.
        """
        tail = self.inner[at:]
        self.inner = self.inner[:at]
        return Vec(*tail)

    @classmethod
    def full(cls, value: T, n: int) -> Vec[T]:
        """
        Initializes a new `Vec` with `n` copies of `value`.

        :param value: The value to fill the `Vec` with.
        :param n: The number of copies of `value` to fill the `Vec` with.
        :returns: A new `Vec` with `n` copies of `value`.
        """
        return Vec[T](*(value for _ in range(n)))

    def push(self, item: T, /) -> None:
        """
        Appends an item to the end of the `Vec`.

        :param item: The item to append.
        """
        self.inner.append(item)

    def clear(self) -> None:
        """
        Removes all elements from the `Vec`.
        """
        self.inner.clear()

    @override
    def extend(self, items: Iterable[T], /) -> None:
        """
        Extends the `Vec` with the elements from the given iterable.

        :param items: The iterable of items to extend the `Vec` with.
        """
        self.inner.extend(items)

    def append(self, other: Self, /):
        """
        Appends the elements of another `Vec` to this `Vec`, then clears the other `Vec`.

        :param other: The `Vec` to append.
        """
        self.inner.extend(other.iter())
        other.clear()

    def first(self) -> Option[T]:
        """
        Returns the first element of the `Vec`, or `Null` if the `Vec` is empty.

        :returns: The first element of the `Vec`, or `Null` if the `Vec` is empty.
        """
        return Null() if self.is_empty() else Some(self[0])

    def last(self) -> Option[T]:
        """
        Returns the last element of the `Vec`, or `Null` if the `Vec` is empty.

        :returns: The last element of the `Vec`, or `Null` if the `Vec` is empty.
        """
        return Null() if self.is_empty() else Some(self[-1])

    def sort_by(
        self: Vec[T],
        compare: Fn1[T, PartialOrd],
        /,
        reverse: bool = False,
    ) -> None:
        """
        Sorts the `Vec` in place using the given comparison function.

        :param compare: The comparison function to use for sorting.
        :param reverse: If `True`, the `Vec` is sorted in reverse order.
        """
        self.inner.sort(key=compare, reverse=reverse)

    def sort[U: PartialOrd](
        self: Vec[U],
        reverse: bool = False,
    ) -> None:
        """
        Sorts the `Vec` in place.

        :param reverse: If `True`, the `Vec` is sorted in reverse order.
        """
        self.inner.sort(reverse=reverse)

    def pop(self) -> Option[T]:
        """
        Pop the last element from the vector and return it.
        :returns: `Some(x)`, where `x` is the last element of the vector if it is not empty,
            otherwise `Null`.
        """
        if self.is_empty():
            return Null()
        return Some(self.inner.pop())

    def remove(self, index: int) -> T:
        """
        Removes and returns the element at position `index` within the vector, shifting all
        elements after it to the left.

        :param index: The index of the element to remove.
        :returns: The removed element previously at index `index` of the vector.
        :raises IndexError: if `index` is out of bounds.
        """
        if index >= self.len():
            raise IndexError(f"removal index (is {index}) should be < len (is {self.len()})")
        return self.inner.pop(index)

    def insert(self, index: int, element: T) -> None:
        """
        Inserts an element at position `index` within the vector, shift all elements after it to
        the right.

        :param index: The index at which to insert the new `element`.
        :param element: The new element to insert at index `index`.
        """
        self.inner.insert(index, element)

    @override
    @classmethod
    def from_iter[P: (True_, False_)](cls, s: Iterator[T, P], /) -> Vec[T]:
        """
        Creates a new `Vec` from an iterator.

        :param s: The iterator to create the `Vec` from.
        :returns: A new `Vec` containing the elements from the iterator.
        """
        return Vec(*s)

    @override
    def clone(self) -> Self:
        """
        Creates a clone of the `Vec`.

        :returns: A clone of the `Vec`.
        """
        return super().clone()

    def retain(self, f: Fn1[T, bool], /) -> None:
        """
        Retains only the elements specified by the predicate function.

        :param f: The predicate function to determine which elements to retain.
        """
        self.inner = list(filter(f, self.inner))

    def choose(self, rng: Rng) -> Option[T]:
        """
        Emulates `SliceRandom::choose` from the `rand` crate.

        Randomly selects an element from the vector and returns it.
        This returns :class:`~Null` if the vector is empty.

        :param rng: The random number generator to use to choose the element.
        :returns: `Some(x)`, where `x` is the randomly-chosen element if the vector is not empty,
            otherwise `Null`.
        """
        if self.is_empty():
            return Null()
        return Some(rng.choice(self.inner))

    def choose_multiple(self, rng: Rng, amount: int) -> Iter[T, False_]:
        """
        Emulates `SliceRandom::choose_multiple` from the `rand` crate.

        Randomly selects `amount` elements - or all elements if `amount` exceeds the
        vector's length - from the vector without replacement and returns an iterator over them.

        :param rng: The random number generator to use to sample the elements without replacement.
        :param amount: The number of elements to sample without replacement,
            capped at the length of the vector.
        :returns: An iterator over the chosen elements.
        """
        # cap the sample size at the population size
        amount = min(amount, self.len())
        # sample without replacement
        sampled_indices = rng.sample(range(self.len()), k=amount)
        return Iter.new(SampledIndexable.new(self, sampled_indices), par=False)

    @qmark
    def choose_multiple_weighted(
        self,
        rng: Rng,
        amount: int,
        weight: Fn1[T, float],
    ) -> Result[Iter[T, False_], ValueError]:
        """
        Similar to :meth:`~Vec.choose_multiple`, but where the likelihood of each element’s
        inclusion in the output may be specified. The elements are returned in an arbitrary,
        unspecified order.

        :param rng: The random number generator to use to sample the elements without replacement.

        :param amount: The number of elements to sample without replacement,
            capped at the length of the vector.

        :param weight: Weighting function mapping each item :math:`x` to a relative likelihood
            :math:`weight(x)`. The probability of each item being selected is thus
            :math:`weight(x) / s`, where :math:`s` is the sum of all :math:`weight(x)`.

        :returns: An iterator over the chosen elements.
        """

        def call(x: T) -> float:
            match weight(x):
                case p if (0 <= p) and math.isfinite(p):
                    return p
                case _:
                    raise ErrShortCircuit(ValueError("Invalid weight"))

        weights = self.iter().map(call).collect(Vec[float]).into()
        # cap the sample size at the population size
        amount = min(amount, self.len())
        return Ok(Iter.new(rng.choices(self.inner, weights=weights, k=amount), par=False))

    def shuffle(self, rng: Rng) -> None:
        """
        Shuffles the elements of the `Vec` in place using the given random number generator.

        :param rng: The random number generator to use for shuffling.
        """
        rng.shuffle(self.inner)

    def fill(self, value: T, /) -> None:
        """
        Fills the `Vec` with the given value.

        :param value: The value to fill the `Vec` with.
        """
        self.inner = self.len() * [value]

    def fill_with(self, f: Fn0[T], /) -> None:
        """
        Fills the `Vec` with values generated by the given function.

        :param f: The function to generate values to fill the `Vec` with.
        """
        self.inner = [f() for _ in range(self.len())]

    def join[U: str](self: Vec[U], separator: U, /) -> U:
        """
        Joins the elements of the `Vec` into a string, separated by the given separator.

        :param separator: The separator to use between elements.
        :returns: A string with the elements of the `Vec` joined by `separator`.
        """
        return separator.join(self.inner)

    def repeat(self, n: int, /) -> Vec[T]:
        """
        Repeats the elements of the `Vec` `n` times.

        :param n: The number of times to repeat the elements.
        :returns: A new `Vec` with the elements repeated `n` times.
        """
        return Vec(*(self.inner * n))

    def flatten[U](self: Vec[Vec[U]]) -> Vec[U]:
        """
        Flattens a `Vec` of `Vec`s into a single `Vec`.

        :returns: A new `Vec` with the elements of the inner `Vec`s.
        """
        return Vec(*(inner for outer in self for inner in outer))

    def dedup(self) -> None:
        """
        Removes consecutive duplicate elements from the `Vec`.

        This method retains only the first occurrence of each group of consecutive duplicate elements.

        :returns: None
        """
        # iterator to start from the second (first) element
        next_iter = self.iter()
        # offset the iterator by one. If there is no second element, return `Null`, otherwise
        # proceed with the deduplication algorithm.
        match next_iter.advance_by(1):
            case Ok(_):

                def not_repeated(pair: tuple[T, T], /) -> bool:
                    return pair[0] != pair[1]

                # iterator starting from the first (zeroth) element
                prev_iter = self.iter()
                self.inner = (
                    prev_iter.zip_longest(next_iter)
                    # retain paris with unequal elements
                    .filter(not_repeated)
                    # flatten by taking the first element of each successive pair.
                    .map(lambda x: x[0])
                    .collect(Vec[T])
                    .into()
                )
            case Err(_):
                return None


if TESTING:
    import math

    def test_sort():
        vec = Vec[int](3, 2, 1)
        vec.sort()
        assert vec == Vec(1, 2, 3)
        vec.sort(reverse=True)
        assert vec == Vec(3, 2, 1)
        vec.sort_by(lambda x: -x, reverse=True)
        assert vec == Vec(1, 2, 3)

    def test_filter():
        vec = Vec[int](1, 2, 3, 6)
        evens = vec.iter().filter(lambda x: x % 2 == 0).collect(Vec[int])
        assert evens == Vec(2, 6)

    def test_chain():
        vec = Vec[int](1, 2, 3)
        vec2 = Vec[int](2, 3, 4)
        res: Vec[int] = vec.iter().chain(vec2.iter()).collect(Vec)
        assert res == Vec(1, 2, 3, 2, 3, 4)

    def test_take():
        vec = Vec[float].full(math.pi, 3)
        taken = vec.iter().take(2).collect(Vec[float])
        assert len(taken) == 2
        taken = vec.iter().take(5).collect(Vec[float])
        assert len(taken) == vec.len()

    def test_take_while():
        vec = Vec[float](*range(5))
        lt5 = vec.iter().take_while(lambda x: x < 5).collect(Vec[float])
        assert len(lt5) == vec.len()
        lt3 = vec.iter().take_while(lambda x: x < 3).collect(Vec[float])
        assert len(lt3) == 3
        vec2 = Vec[Vec[int]](Vec(1, 2, 3))
        ls = [2, 3, 4]
        vec.extend(ls)
        _ = vec2.clone()

    def test_par_iter():
        vec = Vec[float].full(3.14, n=3)
        res = vec.par_iter().take(2).collect(Vec[float])
        assert len(res) == 2

    def test_agg():
        vec = Vec[float](1.0, 2.0, 3.0, -3.0)
        sum_ = vec.iter().sum()
        assert sum_ == 3.0
        prod = vec.iter().product()
        assert prod == -18.0
        max_ = vec.iter().max()
        assert max_ == 3.0

    def test_chunking():
        vec = Vec(*range(97))
        chunks = vec.iter().array_chunk(10)
        assert chunks.count() == math.ceil(100 / 10)
        chunks = vec.par_iter().array_chunk(10)
        assert chunks.count() == math.ceil(100 / 10)

    def test_rng():
        vec = Vec(0, 1, 2, 5, 7, 12, 19)
        rng = Rng(42)
        _ = vec.choose(rng)
        chosen = vec.choose_multiple(rng, 4)
        assert chosen.count() == 4
        chosen = vec.choose_multiple_weighted(rng, 4, lambda _: 1 / vec.len())
        assert chosen.unwrap().count() == 4
        # should short circuit upon violation of the non-negativity constraint
        assert vec.choose_multiple_weighted(rng, 4, lambda _: -1).is_err()

    def test_dedup():
        vec = Vec(1, 1, 3, 2, 2, 2, 4, 5, 4, 4)
        vec.dedup()
        assert vec == Vec(1, 3, 2, 4, 5, 4)

    def test_join():
        vec = Vec("foo", "bar", "baz")
        assert vec.join(",") == "foo,bar,baz"
