import functools
from typing import Callable, cast, overload

from serox.conftest import TESTING

from .option import Null, NullShortCircuit, Option, Some
from .result import Err, ErrShortCircuit, Ok, Result

__all__ = [
    "qmark",
]


@overload
def qmark[T, E, **P](
    f: Callable[P, Result[T, E]],
) -> Callable[P, Result[T, E]]: ...
@overload
def qmark[T, **P](
    f: Callable[P, Option[T]],
) -> Callable[P, Option[T]]: ...


def qmark[T, E, **P](
    f: Callable[P, Result[T, E]] | Callable[P, Option[T]],
) -> Callable[P, Result[T, E]] | Callable[P, Option[T]]:
    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Result[T, E] | Option[T]:
        try:
            return f(*args, **kwargs)
        except ErrShortCircuit as err:  # pyright: ignore[reportUnknownVariableType]
            err = cast(ErrShortCircuit[T, E], err)
            return err.into()
        except NullShortCircuit:
            return Null[T]()

    return wrapper  # pyright: ignore[reportReturnType]


if TESTING:
    import pytest

    def test_qmark():
        @qmark
        def res_fn(s: Result[str, str]) -> Result[str, str]:
            return Ok(s.q)

        err = Err[str, str]("Error")
        assert res_fn(err) == err
        ok = Ok[str, str]("foo")
        assert res_fn(ok) == ok

        @qmark
        def opt_fn(s: Option[str]) -> Option[str]:
            return Some(s.q)

        null = Null[str]()
        assert opt_fn(null) == null
        some = Some("foo")
        assert opt_fn(some) == some

        def opt_fn_no_dec(s: Option[str]) -> Option[str]:
            return Some(s.q)

        null = Null[str]()
        with pytest.raises(NullShortCircuit):
            assert opt_fn_no_dec(null) == null

        @qmark  # pyright: ignore[reportCallIssue, reportArgumentType, reportUntypedFunctionDecorator]
        def opt_fn_bad_sig(_: str) -> str: ...

        del opt_fn_bad_sig
