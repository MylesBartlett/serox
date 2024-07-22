from typing import Protocol, override


class Debug(Protocol):
    @override
    def __repr__(self) -> str: ...
