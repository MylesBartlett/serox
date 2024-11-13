# Serox

_Rusty abstractions for Python_

`serox` provides a suite of commonly-used Rust abstractions in a manner that is near-fully
static-type-checker compliant, the exceptions being cases involving higher-kinded types (HKTs; e.g.
`Iterator.collect`) as these are not currently supported by Python's type system.

The subset of abstractions most broadly-applicable are importable from `serox.prelude`.

## Features

1. `Iterator` combinators that allow for the seamless chaining of operations over data with
   [rayon]-inspired functionality for effortless parallelism.
2. A `Result` pseudo-`enum` comprising `Some` and `Null` pseudo-variants. We say 'pseudo' as the
   Python analogue to Rust's tagged union is the union (`A | B`) type; since this type is not a data
   structure, we cannot implement methods on it directly and instead have to resort to some
   legerdemain.

3. An `Option` pseudo-`enum`. The `T | None` pattern is ubiquitous in Python yet, frustratingly, is
   not treated as a first-class citizen within the language; `Option` is a drop-in replacement that
   redresses this.

4. The `qmark` decorator emulates the '?' (error/null short-circuiting) operator, allowing for
   propagation of error and null values without disrupting the control flow. Without this, one has
   to resort to awkward pattern-matching to perform common operations such as `unwrap_or` (setting
   `Null` to a default value) or `map` (applying a function to the contained value if `Some`).

## Example

Early exiting (in the fashion of Rust's `?` operator) an Option/Result-returning function is enabled
by the `qmark` ('question mark') decorator:

```python
from serox.prelude import *

@qmark
def some_function(value: Option[int]) -> Option[float]:
    squared: int = value.map(lambda x: x ** 2).q
    # The above expands to the rather verbose:
    # match value:
    #     case Null():
    #         return Null[float]()
    #     case Some(x):
    #         squared = value ** 2

    return Some(1.0 / squared)
```

## Requirements

Python version `>=3.12.3` is required for typing purposes.

## Installation

`serox` is available on PyPI and thus the latest version can be installed via `pip` with

```sh
pip install serox
```

or via [uv] with

```sh
uv add serox
```

## Acknowledgements

Credit to [result](https://github.com/rustedpy/result) and
[rustshed](https://github.com/pawelrubin/rustshed/) for laying the groundwork for the `Result` and
`qmark` implementations.

[rayon]: https://github.com/rayon-rs/rayon
[uv]: https://docs.astral.sh/uv/
