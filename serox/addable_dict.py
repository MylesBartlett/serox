from typing import Hashable, Self, override

from serox import Addable, SelfAddable
from serox.misc import Clone

__all__ = ["AddDict"]


class AddDict[_K: Hashable, _V: SelfAddable](
    dict[_K, _V],
    Addable[int | dict[_K, _V]],
    Clone,
):
    """
    Extension of the built-in dictionary class that supports the use of the ``__add__`` operator for
    key-wise addition.

    :example:
        .. code-block:: python

            # Simple case of addition of integers.
            d1 = AddDict({"foo": 1, "bar": 2})
            d2 = {"foo": 3, "bar": 4}
            d1 + d2  # {'foo': 4, 'bar': 6}

            # Concatenation of lists
            d3 = AddDict({"foo": [1], "bar": [2]})
            d4 = {"foo": [3, 4], "bar": 4}
            d3 + d4  # {'foo': [1, 3, 4], 'bar': [2, 4]}
    """

    @override
    def __add__(self, other: int | dict[_K, _V]) -> Self:
        # Allow ``other`` to be an integer, but specifying the identity function, for compatibility
        # with the 'no-default' version of``sum``.
        if isinstance(other, int):
            return self
        cloned = self.clone()

        for key_o, value_o in other.items():
            if key_o in self:
                value_s = self[key_o]
                cloned[key_o] = value_s + value_o
            else:
                cloned[key_o] = value_o
        return cloned

    def __radd__(self, other: int | dict[_K, _V]) -> Self:
        return self + other

    @override
    def __setitem__(self, __key: _K, __value: _V) -> None:
        return super().__setitem__(__key, __value)
