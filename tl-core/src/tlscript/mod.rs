//! `.tlscript` frontend building blocks.
//!
//! This module hosts the zero-copy lexer/token layer for the Tileline script language.
//! It is intentionally RAM-only and accepts in-memory `&str` sources to avoid any disk I/O
//! coupling in the hot path.

pub mod lexer;
pub mod token;

pub use lexer::{LexError, LexErrorKind, Lexer};
pub use token::{Span, Token, TokenKind};
