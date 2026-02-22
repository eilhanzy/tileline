# `.tlscript` Semantic Analyzer (Type Safety, Handles, WASM Sandboxing)

This document describes the semantic validation stage for `.tlscript` after parsing and before
WASM code generation. The semantic analyzer validates types, exported function ABI compatibility,
handle-safe usage, and WASM sandbox restrictions.

## Implementation Location

- `tl-core/src/tlscript/semantic.rs`
- public exports: `tl-core/src/tlscript/mod.rs`, `tl-core/src/lib.rs`

## Goals

- Strong static typing with inference (`let x = 5` -> `int`)
- Zero runtime type checks for normal script execution paths
- Soft diagnostics (warnings + errors) without hard panics in engine-facing semantic flow
- Handle-based memory safety (no raw pointers)
- WASM sandbox safety before codegen/runtime submission
- Borrowing-friendly analysis (avoid unnecessary AST duplication)

## Input / Output

### Input

- Parsed AST (`Module<'src>`) from the `.tlscript` parser

### Output

- `SemanticReport<'src>`: validated signatures, function summaries, safety posture
- `SemanticOutcome<'src>`: soft diagnostics wrapper (`errors`, `warnings`, `can_codegen`)

## Strong Inference and Type Safety

The analyzer performs static type inference and enforcement for the V1 type system:

- `int`
- `float`
- `bool`
- `str`
- `handle` (engine-managed opaque handle, semantic-only type)
- `unit` (implicit no-return/no-value)

### Inference Rules (Current)

- `let x = 5` infers `int`
- `let x = 1.0` infers `float`
- `let x = true` infers `bool`
- `let x = "name"` infers `str`
- handle-acquire intrinsic calls (for example `spawn_sprite()`) infer `handle`

If a declaration has an explicit annotation, the initializer must match at compile time.

### Compile-Time Type Checks

The analyzer currently validates:

- `let` initializer vs annotation
- assignment compatibility
- condition expressions for `if` / `while` (`bool` required)
- `range(...)` arguments (`int` only)
- binary/unary operator operand compatibility
- function call argument compatibility (when callee signature is known)

This is designed to remove runtime type dispatch for normal generated code paths.

## Soft Error Handling ("Soft Exceptions")

The semantic layer provides two APIs:

- `analyze(...)` -> strict `Result<SemanticReport, SemanticError>`
- `analyze_soft(...)` -> `SemanticOutcome` (best-effort report + diagnostics)

`analyze_soft(...)` is the engine/editor-friendly path and supports graceful continuation:

- emits detailed errors and warnings
- returns partial summaries for functions that validated successfully
- sets `can_codegen = false` when codegen must be blocked

This avoids hard `panic!` behavior in semantic validation while preserving precise diagnostics.

### Warnings vs Errors

Examples of warnings:

- implicit type inference used (`let x = ...`)
- discarded expression values
- unannotated non-export parameter

Examples of fatal semantic errors:

- type mismatch
- unknown symbol
- invalid `@export` ABI
- pointer/raw-memory intrinsic usage
- handle use-after-release

## Handle-Based Memory Safety (No Raw Pointers)

`.tlscript` does not allow raw pointer semantics. Engine resources are expected to flow through
opaque handles (sprite/body/etc.) managed by engine intrinsics.

### Semantic Guarantees (V1)

- raw pointer-like intrinsics are rejected (`raw_ptr`, `ptr_*`, `addr_of`, etc.)
- handle release calls are validated as standalone statements
- releasing non-handle values is rejected
- double release is rejected
- use-after-release is rejected

### Handle Acquire / Release Intrinsics

The analyzer uses config allowlists:

- `handle_acquire_call_allowlist`
- `handle_release_call_allowlist`

This allows engine/runtime integration to standardize safe resource APIs without exposing pointers.

## `@export` Bridge and WASM ABI Validation

Functions marked with `@export` are intended for the engine bridge (MPS/WASM task entrypoints), so
the semantic analyzer validates their signatures under an ABI policy (`ExportAbiPolicy`).

### Current ABI Policy (Default)

- typed params required
- `str` disallowed by default
- `handle` disallowed by default
- scalar types allowed: `int`, `float`, `bool`
- `unit` return allowed

This keeps exported signatures deterministic and WASM-friendly while avoiding engine object pointer
leaks across the boundary.

## WASM Sandboxing Rules

Before codegen, the semantic analyzer blocks unsafe or non-portable script calls:

- pointer-like intrinsics (`raw_ptr`, `ptr_*`, `addr_of`, etc.)
- raw WASM memory/table intrinsics (`memory_grow`, `memory_copy`, etc.)
- unknown external calls (unless explicitly allowlisted)

This keeps script behavior compatible with sandboxed execution in the MPS-hosted WASM runtime.

## Ownership, Lifetimes, and Scope Behavior

`.tlscript` currently uses an owned-only lexical scope model:

- no borrow/reference syntax
- variables are scoped to blocks
- identifiers cannot be used outside the scope where they are defined
- released handles are tracked and cannot be used again

This policy is represented explicitly in the semantic report as:

- `OwnershipLifetimePolicy::OwnedOnlyLexicalScopes`

## Bounds Control Policy

The semantic analyzer reports `BoundsCheckPolicy::Required`.

### Current V1 Enforcement Strategy

Indexing syntax is not yet exposed in the V1 parser/AST, so unchecked indexing is not representable.
The analyzer reports:

- `BoundsCheckEnforcement::GuaranteedBySurfaceSyntaxV1`

When indexing/slices are added later, this should evolve into explicit checked-index validation.

## Performance Notes

- Semantic analysis walks AST nodes by reference (no deep AST cloning)
- Borrowed AST names/literals remain `&str`
- Scope tracking is lightweight hash maps per lexical scope
- Diagnostics are accumulated in vectors (`analyze_soft`) for engine/editor consumption

## Current Limitations (V1)

- No return statements yet (parser/semantic model can be extended)
- No user-defined types or generics
- No array/list indexing syntax (bounds policy is preparatory)
- No effect system (purity/async/side-effect classification deferred)
- Handle detection is allowlist-based (not a formal type syntax yet)

## Example: Soft Semantic Outcome

Typical engine/editor integration should prefer:

1. Parse source to AST
2. Run `SemanticAnalyzer::analyze_soft(...)`
3. Log `errors`/`warnings` to engine console
4. If `can_codegen`, continue to IR/WASM codegen
5. Otherwise skip codegen for that script and keep the game loop running

This is the intended "soft exception" behavior for scripting integration.

