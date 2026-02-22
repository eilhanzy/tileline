//! Token definitions for `.tlscript`.
//!
//! The token model is designed for zero-copy lexing:
//! identifiers and literal slices borrow directly from the original source buffer.

/// 1-based source location span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// Byte offset into the source where the token starts.
    pub start: usize,
    /// Byte offset into the source where the token ends (exclusive).
    pub end: usize,
    /// 1-based line number of the token start.
    pub line: u32,
    /// 1-based column number of the token start.
    pub column: u32,
}

impl Span {
    /// Create a new span.
    #[inline]
    pub const fn new(start: usize, end: usize, line: u32, column: u32) -> Self {
        Self {
            start,
            end,
            line,
            column,
        }
    }
}

/// Token kind for `.tlscript`.
///
/// All borrowed variants keep references to the original source string.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenKind<'src> {
    // Structural / virtual tokens
    /// Line break token emitted for non-carriage-return line endings.
    Newline,
    /// Virtual indentation increase token (Python-like block start).
    Indent,
    /// Virtual indentation decrease token (Python-like block end).
    Dedent,
    /// End-of-input token (emitted once after trailing dedents).
    Eof,

    // Keywords / directives
    Def,
    If,
    Elif,
    Else,
    While,
    For,
    In,
    Let,
    TypeInt,
    TypeFloat,
    TypeBool,
    TypeStr,
    /// `true`
    True,
    /// `false`
    False,
    /// `@export`
    ExportDecorator,
    /// Any decorator other than `@export`, without the `@` prefix.
    Decorator(&'src str),

    // Identifiers / literals
    Identifier(&'src str),
    Integer(&'src str),
    Float(&'src str),
    /// Raw inner string slice without the surrounding quotes.
    ///
    /// Escapes are not decoded at lexing time to preserve zero-copy behavior.
    String(&'src str),

    // Punctuation / operators
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Colon,
    Comma,
    Dot,
    At,
    Arrow,
    Assign,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    EqEq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    AndAnd,
    OrOr,
}

/// A token plus its source span.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Token<'src> {
    /// Token classification.
    pub kind: TokenKind<'src>,
    /// Source location span.
    pub span: Span,
}

impl<'src> Token<'src> {
    /// Create a new token.
    #[inline]
    pub const fn new(kind: TokenKind<'src>, span: Span) -> Self {
        Self { kind, span }
    }
}
