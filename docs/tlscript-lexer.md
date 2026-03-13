# `.tlscript` Zero-Copy Lexer and Token Model

This document describes the first frontend layer of Tileline's scripting pipeline: tokenization for
`.tlscript` (Python-like syntax, MPS-targeted WASM execution path).

## Goals

- Zero-copy tokenization from an in-memory `&str`
- No `std::fs` dependency in the lexer hot path
- Python-like indentation semantics via virtual `Indent` / `Dedent` tokens
- Predictable, cache-friendly scanning for frame-budget-sensitive tooling/runtime use

## Implementation Location

- `tl-core/src/tlscript/token.rs`
- `tl-core/src/tlscript/lexer.rs`
- `tl-core/src/tlscript/mod.rs`

## Token Design

`TokenKind<'src>` stores borrowed slices for:

- `Identifier(&'src str)`
- `Integer(&'src str)`
- `Float(&'src str)`
- `String(&'src str)` (raw inner slice, no escape decoding)
- `Decorator(&'src str)`

This avoids heap allocations during lexing and keeps source ownership external to the lexer.

### Virtual Structural Tokens

The lexer emits Python-style block structure tokens:

- `Newline`
- `Indent`
- `Dedent`
- `Eof`

`Indent`/`Dedent` are produced from an indentation depth stack and do not correspond to literal
source substrings.

## Indentation Semantics

The lexer tracks indentation only at line start:

- Leading spaces increase indentation depth
- Returning to a previous depth emits one or more `Dedent`
- Unknown indentation levels produce `InvalidDedent`
- Leading tabs are rejected (`TabIndentation`) to keep block depth deterministic

Inline tabs/spaces (after the first non-whitespace token on a line) are treated as normal spacing.

## Supported Keywords / Directives (Current)

Control flow and declarations:

- `def`
- `if`
- `elif`
- `else`
- `while`
- `for`
- `in`
- `let`

Primitive types:

- `int`
- `float`
- `bool`
- `str`

Literals/directives:

- `true`
- `false`
- `@export` (`ExportDecorator`)

Other decorators are currently tokenized as `Decorator(name)` without semantic validation.

## Error Model

`Lexer` yields `Result<Token<'src>, LexError>` and stops after the first error.

Current error categories:

- `TabIndentation`
- `InvalidDedent`
- `InvalidDecorator`
- `UnterminatedString`
- `UnexpectedByte`
- `InvalidOperator`

Errors carry a `Span` with byte offsets and 1-based line/column coordinates.

## Performance Notes

- Byte-slice scanning (`&[u8]`) is used for predictable hot-path branching.
- Token payloads borrow from the original `&str`.
- The indentation stack preallocates capacity and grows only on deeper nesting.
- No locks, channels, or heap allocations are required per token in the common path.

## Known V1 Limitations

- No escape decoding for string literals (parser/later passes may normalize if needed)
- No triple-quoted/multiline strings
- Parser and semantic analyzer now exist; this document covers lexer/token only
- No WASM codegen in this stage

## Next Steps

1. Parser with indentation-aware block construction
2. Typed AST / semantic validation
3. IR + WASM codegen
4. MPS integration (`WasmTask`) and host ABI binding (`wit-bindgen`)
