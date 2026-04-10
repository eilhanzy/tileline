# Contributing to Tileline

Thanks for your interest in contributing.

Tileline is currently in a pre-alpha transition phase, and `v0.5.0` is the main stabilization
milestone before broad public growth. Contributions are still welcome now, especially if they are
well-scoped and test-backed.

## Before You Start

- Read [README.md](README.md) for workspace context.
- Read [docs/README.md](docs/README.md) for subsystem docs.
- Check open issues first to avoid duplicate work.

For large changes (new subsystems, major API shifts, broad refactors), open an issue first and
align on scope before implementation.

## Development Setup

Requirements:
- Rust toolchain (stable)
- Cargo

Recommended baseline checks:

```bash
cargo check
cargo test -p paradoxpe
cargo check -p runtime
```

If your change touches other crates, run targeted checks/tests for those crates too.

## What We Need Most (Current Priority)

While `v0.5.0` is in progress, high-value contributions are:

- bug fixes with reproducible steps
- targeted performance regressions with telemetry evidence
- docs improvements and onboarding clarity
- tests for runtime/physics/script edge cases
- small, isolated quality improvements that do not destabilize release scope

## Pull Request Guidelines

- Keep PRs focused and small enough to review quickly.
- Include problem statement, approach, and verification steps.
- Add or update tests when behavior changes.
- Update docs when public behavior, workflows, or commands change.
- Avoid unrelated cleanup in the same PR.

PR checklist:

- [ ] I linked an issue (or explained why no issue is needed).
- [ ] I ran relevant checks/tests locally.
- [ ] I updated docs for user-facing changes.
- [ ] I included clear reproduction/validation steps.

## Commit and Branch Hygiene

- Use clear, descriptive commit messages.
- Prefer one logical topic per PR.
- Do not force-push over reviewer discussion without noting what changed.

## Review Expectations

The team aims to provide an initial response as quickly as possible. For complex PRs, review may
be staged (architecture first, then implementation details).

## Communication

- Use GitHub Issues for bugs and proposals.
- Use Pull Requests for concrete code/doc changes.
- Keep discussions technical, concise, and reproducible.

## Code of Conduct

By participating, you agree to follow [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
