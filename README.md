# Serox: Serpentine Oxidation :fire:

Rusty Abstractions for Python.

`Serox` defines a emulates a suite of commonly-used Rust primitives. Namely:

1. Iterator combinators allowing for the seamless chaining of operations.
2. A `Result` pseudo-`enum` comprising `Some` and `Null` pseudo-variants. We say 'pseudo' as the
   Python analogue to Rust's tagged union is the union (`A | B`) type; since this type is not a data
   structure, we cannot implement methods on it directly and instead have to resort to some
   legerdemain.

3. A `Option` pseudo-`enum`. The `T | None` pattern is ubiquitous in Python yet, frustratingly, is
   not treated as a first-class citizen within the language; `Option` redresses this.

4. The `qmark` decorator emulates the '?' (error/null short-circuiting) operator, allowing for
   propagation of error and null values without interrupting the control flow. Without this, one has
   to resort to awkward pattern-matching to perform common operations such as `unwrap_or` (setting
   `Null` to a default value) or `map` (applying a function to the contained value if `Some`).

```python
from serox import Option, qmark

@qmark
def some_function(foo: Option[str]) -> Option[str]:
    foo_bar: str = value.map(lambda x: x + "bar").q
    return foo_bar + "_baz"
```

## Acknowledgements

Credit to [result](https://github.com/rustedpy/result) and
[rustshed](https://github.com/pawelrubin/rustshed/) for laying the groundwork for the
[Result](https://github.com/rustedpy/result) and '?' implementations.
