## General

- Keep the README.md and the documentation inside docs/ updated as things change in the code.
- Make sure you have the venv activated, before running anything.
- When the code and the tests disagree, first ask: is this a bug in the code, or are the tests out of date? Fix whichever is wrong. Never make implementation decisions (method names, visibility, signatures, behavior) based on what the tests happen to call — the tests follow the code, not the other way around.

## Imports

- Avoid conditional imports.


## Typing

- Use modern type-hint syntax everywhere: builtin generics (`list[str]`, `dict[str, Any]`, `tuple[int, ...]`) over `typing.List` / `typing.Dict`, `Optional[X]` for nullable types (not `X | None`), and annotate return types on every function — including `-> None` for procedures.


## Code layout

- _private functions/methods should always come after public ones.


## Comments & docstrings

- Do not erase comments, unless they dont make sence anymore.
- Write docstrings for modules, public classes, and non-trivial algorithms only — skip them for self-evident code. Docstrings describe the function's contract; they should not narrate the current implementation, nor reference specific call sites that might move.
- Do not put conversation-specific comments in the code.


## Output

- Use `logging.getLogger("alphazoo")` for output, not `print`.
