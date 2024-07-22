from typing import Any, TypeGuard  # noqa: I001


__all__ = [
    "default_if_none",
    "none",
    "some",
    "unwrap_or",
]


def some[T](value: T | None, /) -> TypeGuard[T]:
    """
    Returns ``True`` if the input is **not** ``None``
    (that is, if the ``Optional`` monad contains some value).

    :param value: Value to be checked.
    :returns: ``True`` if ``value`` is **not** ``None`` else ``False``.
    """
    return value is not None


def none(value: Any | None, /) -> TypeGuard[None]:
    """
    Returns ``True`` if the input **is** ``None``
    (that is, if the ``Optional`` monad contains no value).

    :param value: Value to be checked.
    :returns: ``True`` if ``value`` **is** ``None`` else ``False``.
    """
    return value is None


def unwrap_or[T](value: T | None, /, default: T) -> T:
    """
    Returns the input if the input is **not** None else the specified
    ``default`` value.

    :param value: Input to be unwrapped and returned if not ``None``.
    :param default: Default value to use if ``value`` is ``None``.
    :returns: ``default`` if ``value`` is ``None`` otherwise ``value``.
    """
    return default if value is None else value


default_if_none = unwrap_or
