# AGENTS.md

These instructions define the coding standards to follow in this repository,
with a focus on Python code that is functional, easy to parse by a human, and
type-safe.

## Core Principles
- Prefer small, pure functions with explicit inputs and outputs.
- Keep functions focused; avoid hidden side effects and global state.
- Make code easy to read: clear naming, linear control flow, minimal nesting.
- Use static typing and type-driven design to clarify intent and catch errors.

## Function Design
- Write functions that do one thing and return values instead of mutating state.
- Pass dependencies as parameters rather than reaching into globals.
- Use dataclasses or TypedDict only when they improve clarity over primitives.
- Keep functions short; split logic into helpers when a function grows complex.

## Types and Annotations
- Add type hints for all public functions, methods, and module-level constants.
- Use standard library types from `typing` (e.g., `Iterable`, `Sequence`).
- Prefer precise types over `Any`. Use `Any` only when strictly necessary.
- Use `Optional[T]` instead of nullable comments; avoid implicit `None`.
- When returning multiple values, use a small `NamedTuple` or dataclass.

## Readability and Structure
- Keep module layout predictable: imports, constants, types, functions, main.
- Use docstrings for non-obvious behavior or invariants, not for the obvious.
- Prefer explicit variable names over abbreviated ones, except for indices.
- Use guard clauses to reduce nesting; avoid long `if/else` chains.

## Error Handling
- Validate inputs early and fail fast with clear exceptions.
- Avoid bare `except`; catch specific exceptions and re-raise with context.
- Use `Result`-style return values only when error flow is expected and simple.

## Style and Tooling
- Follow PEP 8; keep lines <= 88 characters when practical.
- Prefer f-strings for formatting.
- Avoid cleverness; clarity beats brevity.
