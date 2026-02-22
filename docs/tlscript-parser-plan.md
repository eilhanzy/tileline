# `.tlscript` Parser / AST Plan (V1)

This document defines the planned parser architecture for `.tlscript` after the zero-copy lexer
stage. The goal is a Python-like syntax with deterministic, engine-friendly semantics that compile
to WASM for execution on MPS.

## Scope (V1)

Planned parser coverage:

- function definitions (`def`)
- decorator prefix handling (`@export`)
- indentation-based blocks (`Indent` / `Dedent`)
- `if / elif / else`
- `while`
- `for ... in range(...)` (initially constrained form)
- `let` bindings
- assignment (`=`)
- function calls
- typed parameters / local type annotations (`int`, `float`, `bool`, `str`)
- arithmetic/comparison/logical expressions

Intentionally deferred:

- classes
- imports/modules
- exceptions
- comprehensions
- pattern matching

## Inputs and Outputs

### Input

- `tl_core::Lexer<'src>` token stream (`Token<'src>`)
- zero-copy token payloads referencing the original source

### Output

- AST that borrows from source (`&'src str`) where practical
- parse diagnostics with spans

This keeps parser allocations limited to AST nodes/collections, not duplicated strings.

## Parser Strategy

## 1. Top-Level Structure

Use a recursive-descent parser with Pratt parsing (or precedence climbing) for expressions:

- Recursive descent is a good fit for indentation-driven statements and predictable control flow.
- Pratt/precedence parsing keeps expression parsing compact and fast.

Proposed API shape:

```rust
pub struct Parser<'src, I>
where
    I: Iterator<Item = Result<Token<'src>, LexError>>,
{
    // lookahead + diagnostics + AST state
}

impl<'src, I> Parser<'src, I>
where
    I: Iterator<Item = Result<Token<'src>, LexError>>,
{
    pub fn parse_module(&mut self) -> Result<Module<'src>, ParseError>;
}
```

## 2. Token Consumption Model

- Keep a small lookahead buffer (1-2 tokens is enough for V1)
- Preserve spans on all syntax nodes
- Treat lexer errors as parser input errors and surface them unchanged (or wrapped)
- Skip repeated `Newline` tokens between top-level statements

## 3. Indentation Handling

The lexer already emits virtual `Indent` / `Dedent` tokens. The parser should:

- require `:` then `Newline` then `Indent` for block-starting statements
- parse statements until `Dedent`
- allow blank lines inside blocks (`Newline` only)
- preserve deterministic block boundaries without whitespace rescanning

This is simpler and faster than re-measuring indentation in the parser.

## AST Plan (Borrowing-Friendly)

## Core Nodes

Recommended V1 AST node set (illustrative names):

- `Module<'src>`
- `Item<'src>`
  - `FunctionDef<'src>`
- `Stmt<'src>`
  - `Let`
  - `Assign`
  - `If`
  - `While`
  - `ForRange`
  - `Expr`
  - `Return` (optional V1, recommended to include early)
- `Expr<'src>`
  - `Identifier(&'src str)`
  - `IntLiteral(&'src str)`
  - `FloatLiteral(&'src str)`
  - `BoolLiteral(bool)`
  - `StringLiteral(&'src str)`
  - `Unary`
  - `Binary`
  - `Call`
  - `Grouping`

## Decorators

Decorators should be parsed as metadata attached to `FunctionDef`:

- V1 accepted: `@export`
- other decorators may parse syntactically but fail in semantic validation

This keeps the parser generic while allowing the semantic pass to enforce engine ABI rules.

## Type Annotations

V1 parser should support shallow type syntax only:

- parameter annotations: `name: int`
- local declarations: `let x: float = ...`
- return annotations (optional but recommended): `def f() -> int:`

Represent types as a small enum:

- `TypeName::Int`
- `TypeName::Float`
- `TypeName::Bool`
- `TypeName::Str`

No generic types or user-defined types in V1.

## Expression Grammar (V1)

Suggested precedence (highest to lowest):

1. call / postfix
2. unary (`-`, maybe `!` later)
3. multiplicative (`*`, `/`, `%`)
4. additive (`+`, `-`)
5. comparison (`<`, `<=`, `>`, `>=`)
6. equality (`==`, `!=`)
7. logical and (`&&`)
8. logical or (`||`)
9. assignment (`=`) handled at statement level

Notes:

- Keep assignment out of general expressions in V1 to simplify semantics and WASM lowering.
- `for` initially targets `range(...)` only; parser can store a specific `ForRange` node.

## Statement Parsing Rules (V1)

### Function Definitions

Form:

```text
@export
def update(dt: float) -> int:
    ...
```

Parser responsibilities:

- collect zero or more decorators before `def`
- parse function name and parameter list
- parse optional return type
- parse required indented body

### `if / elif / else`

- `elif` should parse as chained branches in a single `If` node
- `else` body is optional
- each branch requires `:` and an indented block

### `while`

- parse condition expression
- parse required block

### `for ... in range(...)`

V1 recommendation:

- parse only `for <ident> in range(<args>):`
- accept `range(end)` and `range(start, end)` initially
- reject arbitrary iterables in V1

This keeps lowering straightforward for the first WASM backend.

## Error Handling and Recovery

Parser errors should be precise and span-based, but recovery can stay simple in V1.

Recommended recovery points:

- statement boundary (`Newline`)
- block boundary (`Dedent`)
- top-level boundary (`def`, `@export`, `Eof`)

This is sufficient for developer feedback without building a full incremental parser yet.

## Semantic Pass Handoff (Planned)

The parser should not enforce all engine rules. It should hand off to a semantic pass that checks:

- duplicate function names
- `@export` restrictions (signature compatibility / ABI)
- type consistency (`let`, assignments, returns)
- disallowed decorators
- `range(...)` argument validity

## WASM / MPS Integration Expectations

Parser output should be stable and explicit enough for downstream phases:

- AST -> typed AST / IR
- IR -> WASM module
- `@export` functions surfaced to MPS-hosted runtime ABI (`wit-bindgen` path)

The parser does not perform code generation and must remain independent of `wasmer`.

## Performance Notes

- Avoid string copies in AST where token slices already provide `&str`
- Prefer small enums and compact node layouts for cache locality
- Keep parser state single-thread-owned and lock-free
- Separate syntax parsing from semantic validation to keep hot-path branches predictable

## Proposed Next Implementation Steps

1. `tl-core/src/tlscript/ast.rs` (AST node enums/structs)
2. `tl-core/src/tlscript/parser.rs` (recursive descent + Pratt expressions)
3. parser unit tests (indent blocks, decorators, precedence, error spans)
4. semantic/type pass skeleton

