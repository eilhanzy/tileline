//! Zero-copy lexer for `.tlscript`.
//!
//! Design notes:
//! - Accepts only in-memory `&str` input (no `std::fs` coupling).
//! - Emits borrowed token slices referencing the original source.
//! - Tracks indentation depth with a stack and emits virtual `Indent`/`Dedent`.
//! - Uses byte-oriented scanning for cache locality and predictable hot-path behavior.

use std::fmt;

use super::token::{Span, Token, TokenKind};

/// Lexing error kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LexErrorKind {
    /// Indentation mixes an unsupported tab character.
    TabIndentation,
    /// Indentation depth did not match any prior indentation level.
    InvalidDedent,
    /// `@` was not followed by an identifier.
    InvalidDecorator,
    /// String literal was not terminated before newline or EOF.
    UnterminatedString,
    /// An unexpected byte was encountered.
    UnexpectedByte,
    /// A single `&` or `|` was encountered where `&&` / `||` is required.
    InvalidOperator,
}

/// Zero-allocation lexer error object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LexError {
    /// Error category.
    pub kind: LexErrorKind,
    /// Source span for the error location.
    pub span: Span,
}

impl LexError {
    #[inline]
    const fn new(kind: LexErrorKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} at line {}, column {}",
            self.kind, self.span.line, self.span.column
        )
    }
}

impl std::error::Error for LexError {}

/// Zero-copy `.tlscript` lexer.
///
/// The iterator yields `Result<Token<'src>, LexError>` to preserve hot-path scanning while keeping
/// syntax/indentation validation local to the lexer.
pub struct Lexer<'src> {
    src: &'src str,
    bytes: &'src [u8],
    pos: usize,
    line: u32,
    line_start: usize,
    at_line_start: bool,
    emitted_eof: bool,
    pending_indent: bool,
    pending_dedents: u16,
    indent_stack: Vec<usize>,
}

impl<'src> Lexer<'src> {
    /// Construct a lexer from an in-memory source buffer.
    ///
    /// This function performs no I/O and does not copy the source.
    pub fn new(src: &'src str) -> Self {
        let mut indent_stack = Vec::with_capacity(16);
        indent_stack.push(0);
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            line: 1,
            line_start: 0,
            at_line_start: true,
            emitted_eof: false,
            pending_indent: false,
            pending_dedents: 0,
            indent_stack,
        }
    }

    /// Fetch the next token.
    ///
    /// `None` is returned only after the `Eof` token has been emitted.
    pub fn next_token(&mut self) -> Option<Result<Token<'src>, LexError>> {
        if self.emitted_eof {
            return None;
        }

        match self.next_token_inner() {
            Ok(token) => {
                if matches!(token.kind, TokenKind::Eof) {
                    self.emitted_eof = true;
                }
                Some(Ok(token))
            }
            Err(err) => {
                self.emitted_eof = true;
                Some(Err(err))
            }
        }
    }

    fn next_token_inner(&mut self) -> Result<Token<'src>, LexError> {
        if self.pending_indent {
            self.pending_indent = false;
            return Ok(self.virtual_token(TokenKind::Indent));
        }
        if self.pending_dedents > 0 {
            self.pending_dedents -= 1;
            return Ok(self.virtual_token(TokenKind::Dedent));
        }

        loop {
            if self.at_line_start {
                if let Some(token) = self.handle_line_start()? {
                    return Ok(token);
                }
                if self.pending_indent {
                    self.pending_indent = false;
                    return Ok(self.virtual_token(TokenKind::Indent));
                }
                if self.pending_dedents > 0 {
                    self.pending_dedents -= 1;
                    return Ok(self.virtual_token(TokenKind::Dedent));
                }
            }

            if self.pos >= self.bytes.len() {
                if self.indent_stack.len() > 1 {
                    self.indent_stack.pop();
                    return Ok(self.virtual_token(TokenKind::Dedent));
                }
                return Ok(self.virtual_token(TokenKind::Eof));
            }

            let b = self.bytes[self.pos];
            match b {
                b' ' | b'\t' => {
                    // Inline whitespace is skipped. Tabs inside a line are allowed as spacing.
                    self.pos += 1;
                }
                b'#' => {
                    self.skip_comment();
                }
                b'\n' => return Ok(self.consume_newline_token()),
                b'\r' => {
                    if self.peek_byte(1) == Some(b'\n') {
                        self.pos += 1;
                    }
                    return Ok(self.consume_newline_token());
                }
                _ => return self.lex_non_whitespace_token(),
            }
        }
    }

    fn handle_line_start(&mut self) -> Result<Option<Token<'src>>, LexError> {
        loop {
            self.at_line_start = false;
            let line = self.line;
            let mut indent = 0usize;
            let mut cursor = self.pos;

            while let Some(b) = self.bytes.get(cursor).copied() {
                match b {
                    b' ' => {
                        indent += 1;
                        cursor += 1;
                    }
                    b'\t' => {
                        return Err(self.error_at(
                            LexErrorKind::TabIndentation,
                            cursor,
                            cursor + 1,
                            line,
                            (cursor - self.line_start + 1) as u32,
                        ));
                    }
                    _ => break,
                }
            }

            match self.bytes.get(cursor).copied() {
                Some(b'#') => {
                    self.pos = cursor;
                    self.skip_comment();
                    // Silently consume the newline that ends the comment line so that a
                    // comment-only line is fully invisible to the parser (no Newline token
                    // emitted). Without this, the next iteration would return a Newline for
                    // the `\n` after the `#...` content, which breaks block parsing when
                    // comments appear before the first statement of an indented block.
                    match self.bytes.get(self.pos).copied() {
                        Some(b'\r') => {
                            self.pos += 1;
                            if self.bytes.get(self.pos).copied() == Some(b'\n') {
                                self.pos += 1;
                            }
                            self.line = self.line.saturating_add(1);
                            self.line_start = self.pos;
                        }
                        Some(b'\n') => {
                            self.pos += 1;
                            self.line = self.line.saturating_add(1);
                            self.line_start = self.pos;
                        }
                        _ => {}
                    }
                    continue;
                }
                Some(b'\n') => {
                    self.pos = cursor;
                    return Ok(Some(self.consume_newline_token()));
                }
                Some(b'\r') => {
                    self.pos = cursor;
                    if self.peek_byte(1) == Some(b'\n') {
                        self.pos += 1;
                    }
                    return Ok(Some(self.consume_newline_token()));
                }
                None => {
                    self.pos = cursor;
                    return Ok(None);
                }
                Some(_) => {
                    self.pos = cursor;
                    self.apply_indent(indent)?;
                    return Ok(None);
                }
            }
        }
    }

    fn apply_indent(&mut self, indent: usize) -> Result<(), LexError> {
        let current = *self.indent_stack.last().unwrap_or(&0);
        if indent == current {
            return Ok(());
        }

        if indent > current {
            self.indent_stack.push(indent);
            self.pending_indent = true;
            return Ok(());
        }

        let target_index = self.indent_stack.iter().rposition(|&depth| depth == indent);
        let Some(target_index) = target_index else {
            return Err(self.error_here(LexErrorKind::InvalidDedent));
        };

        let dedents = self.indent_stack.len() - 1 - target_index;
        self.indent_stack.truncate(target_index + 1);
        self.pending_dedents = dedents as u16;
        Ok(())
    }

    fn lex_non_whitespace_token(&mut self) -> Result<Token<'src>, LexError> {
        let start = self.pos;
        let line = self.line;
        let column = self.current_column();
        let b = self.bytes[self.pos];

        let kind = match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.lex_identifier_or_keyword(),
            b'0'..=b'9' => self.lex_number(),
            b'"' | b'\'' => self.lex_string()?,
            b'(' => {
                self.pos += 1;
                TokenKind::LParen
            }
            b')' => {
                self.pos += 1;
                TokenKind::RParen
            }
            b'[' => {
                self.pos += 1;
                TokenKind::LBracket
            }
            b']' => {
                self.pos += 1;
                TokenKind::RBracket
            }
            b'{' => {
                self.pos += 1;
                TokenKind::LBrace
            }
            b'}' => {
                self.pos += 1;
                TokenKind::RBrace
            }
            b':' => {
                self.pos += 1;
                TokenKind::Colon
            }
            b',' => {
                self.pos += 1;
                TokenKind::Comma
            }
            b'.' => {
                self.pos += 1;
                TokenKind::Dot
            }
            b'@' => self.lex_decorator_or_at()?,
            b'+' => {
                self.pos += 1;
                TokenKind::Plus
            }
            b'-' => {
                self.pos += 1;
                if self.match_byte(b'>') {
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            b'*' => {
                self.pos += 1;
                TokenKind::Star
            }
            b'/' => {
                self.pos += 1;
                TokenKind::Slash
            }
            b'%' => {
                self.pos += 1;
                TokenKind::Percent
            }
            b'=' => {
                self.pos += 1;
                if self.match_byte(b'=') {
                    TokenKind::EqEq
                } else {
                    TokenKind::Assign
                }
            }
            b'!' => {
                self.pos += 1;
                if self.match_byte(b'=') {
                    TokenKind::NotEq
                } else {
                    return Err(self.error_at(
                        LexErrorKind::UnexpectedByte,
                        start,
                        start + 1,
                        line,
                        column,
                    ));
                }
            }
            b'<' => {
                self.pos += 1;
                if self.match_byte(b'=') {
                    TokenKind::LtEq
                } else {
                    TokenKind::Lt
                }
            }
            b'>' => {
                self.pos += 1;
                if self.match_byte(b'=') {
                    TokenKind::GtEq
                } else {
                    TokenKind::Gt
                }
            }
            b'&' => {
                self.pos += 1;
                if self.match_byte(b'&') {
                    TokenKind::AndAnd
                } else {
                    return Err(self.error_at(
                        LexErrorKind::InvalidOperator,
                        start,
                        self.pos,
                        line,
                        column,
                    ));
                }
            }
            b'|' => {
                self.pos += 1;
                if self.match_byte(b'|') {
                    TokenKind::OrOr
                } else {
                    return Err(self.error_at(
                        LexErrorKind::InvalidOperator,
                        start,
                        self.pos,
                        line,
                        column,
                    ));
                }
            }
            _ => {
                return Err(self.error_at(
                    LexErrorKind::UnexpectedByte,
                    start,
                    start + 1,
                    line,
                    column,
                ));
            }
        };

        Ok(Token::new(kind, Span::new(start, self.pos, line, column)))
    }

    fn lex_identifier_or_keyword(&mut self) -> TokenKind<'src> {
        let start = self.pos;
        self.pos += 1;
        while let Some(b) = self.bytes.get(self.pos).copied() {
            if is_ident_continue(b) {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = &self.src[start..self.pos];
        match text {
            "def" => TokenKind::Def,
            "if" => TokenKind::If,
            "elif" => TokenKind::Elif,
            "else" => TokenKind::Else,
            "while" => TokenKind::While,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "let" => TokenKind::Let,
            "int" => TokenKind::TypeInt,
            "float" => TokenKind::TypeFloat,
            "bool" => TokenKind::TypeBool,
            "str" => TokenKind::TypeStr,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            _ => TokenKind::Identifier(text),
        }
    }

    fn lex_number(&mut self) -> TokenKind<'src> {
        let start = self.pos;
        self.pos += 1;
        while let Some(b) = self.bytes.get(self.pos).copied() {
            if b.is_ascii_digit() || b == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }

        let mut is_float = false;
        if self.peek_byte(0) == Some(b'.') && self.peek_byte(1).is_some_and(|b| b.is_ascii_digit())
        {
            is_float = true;
            self.pos += 1; // consume '.'
            while let Some(b) = self.bytes.get(self.pos).copied() {
                if b.is_ascii_digit() || b == b'_' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }

        let text = &self.src[start..self.pos];
        if is_float {
            TokenKind::Float(text)
        } else {
            TokenKind::Integer(text)
        }
    }

    fn lex_string(&mut self) -> Result<TokenKind<'src>, LexError> {
        let quote = self.bytes[self.pos];
        let quote_pos = self.pos;
        self.pos += 1;
        let content_start = self.pos;

        while let Some(b) = self.bytes.get(self.pos).copied() {
            match b {
                b'\\' => {
                    // Preserve zero-copy behavior by skipping escaped bytes without decoding.
                    self.pos += 1;
                    if self.pos < self.bytes.len() {
                        self.pos += 1;
                    }
                }
                b if b == quote => {
                    let content_end = self.pos;
                    self.pos += 1;
                    return Ok(TokenKind::String(&self.src[content_start..content_end]));
                }
                b'\n' | b'\r' => {
                    return Err(self.error_at(
                        LexErrorKind::UnterminatedString,
                        quote_pos,
                        self.pos,
                        self.line,
                        (quote_pos - self.line_start + 1) as u32,
                    ));
                }
                _ => self.pos += 1,
            }
        }

        Err(self.error_at(
            LexErrorKind::UnterminatedString,
            quote_pos,
            self.pos,
            self.line,
            (quote_pos - self.line_start + 1) as u32,
        ))
    }

    fn lex_decorator_or_at(&mut self) -> Result<TokenKind<'src>, LexError> {
        let start = self.pos;
        self.pos += 1; // consume '@'

        let Some(b) = self.bytes.get(self.pos).copied() else {
            return Ok(TokenKind::At);
        };

        if !is_ident_start(b) {
            return Ok(TokenKind::At);
        }

        let ident_start = self.pos;
        self.pos += 1;
        while let Some(b) = self.bytes.get(self.pos).copied() {
            if is_ident_continue(b) {
                self.pos += 1;
            } else {
                break;
            }
        }

        let name = &self.src[ident_start..self.pos];
        if name == "export" {
            Ok(TokenKind::ExportDecorator)
        } else if name.is_empty() {
            Err(self.error_at(
                LexErrorKind::InvalidDecorator,
                start,
                self.pos,
                self.line,
                (start - self.line_start + 1) as u32,
            ))
        } else {
            Ok(TokenKind::Decorator(name))
        }
    }

    #[inline]
    fn consume_newline_token(&mut self) -> Token<'src> {
        let start = self.pos;
        let line = self.line;
        let column = self.current_column();
        self.pos += 1;
        self.line = self.line.saturating_add(1);
        self.line_start = self.pos;
        self.at_line_start = true;
        Token::new(TokenKind::Newline, Span::new(start, self.pos, line, column))
    }

    #[inline]
    fn skip_comment(&mut self) {
        while let Some(b) = self.bytes.get(self.pos).copied() {
            if b == b'\n' || b == b'\r' {
                break;
            }
            self.pos += 1;
        }
    }

    #[inline]
    fn match_byte(&mut self, expected: u8) -> bool {
        if self.peek_byte(0) == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    #[inline]
    fn peek_byte(&self, rel: usize) -> Option<u8> {
        self.bytes.get(self.pos + rel).copied()
    }

    #[inline]
    fn current_column(&self) -> u32 {
        (self.pos.saturating_sub(self.line_start) + 1) as u32
    }

    #[inline]
    fn virtual_token(&self, kind: TokenKind<'src>) -> Token<'src> {
        let pos = self.pos;
        Token::new(kind, Span::new(pos, pos, self.line, self.current_column()))
    }

    #[inline]
    fn error_here(&self, kind: LexErrorKind) -> LexError {
        LexError::new(
            kind,
            Span::new(self.pos, self.pos, self.line, self.current_column()),
        )
    }

    #[inline]
    fn error_at(
        &self,
        kind: LexErrorKind,
        start: usize,
        end: usize,
        line: u32,
        column: u32,
    ) -> LexError {
        LexError::new(kind, Span::new(start, end, line, column))
    }
}

impl<'src> Iterator for Lexer<'src> {
    type Item = Result<Token<'src>, LexError>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}

#[inline]
fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

#[inline]
fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collect_kinds(src: &str) -> Vec<TokenKind<'_>> {
        Lexer::new(src).map(|t| t.expect("lex ok").kind).collect()
    }

    #[test]
    fn emits_keywords_indent_dedent_and_export() {
        let src = "@export\ndef update():\n    let x: int = 1\n    if true:\n        let y: float = 1.0\n    else:\n        let s: str = \"ok\"\n";

        let kinds = collect_kinds(src);
        assert!(kinds.contains(&TokenKind::ExportDecorator));
        assert!(kinds.contains(&TokenKind::Def));
        assert!(kinds.contains(&TokenKind::Let));
        assert!(kinds.contains(&TokenKind::TypeInt));
        assert!(kinds.contains(&TokenKind::TypeFloat));
        assert!(kinds.contains(&TokenKind::TypeStr));

        let indent_count = kinds
            .iter()
            .filter(|k| matches!(k, TokenKind::Indent))
            .count();
        let dedent_count = kinds
            .iter()
            .filter(|k| matches!(k, TokenKind::Dedent))
            .count();
        assert_eq!(indent_count, dedent_count);
        assert!(matches!(kinds.last(), Some(TokenKind::Eof)));
    }

    #[test]
    fn identifier_and_literal_slices_are_zero_copy() {
        let src = "let player_name = \"m4\"\n";
        let mut lexer = Lexer::new(src);

        let mut ident_ptr = None;
        let mut string_ptr = None;

        while let Some(tok) = lexer.next() {
            let tok = tok.expect("lex ok");
            match tok.kind {
                TokenKind::Identifier(s) if s == "player_name" => ident_ptr = Some(s.as_ptr()),
                TokenKind::String(s) if s == "m4" => string_ptr = Some(s.as_ptr()),
                TokenKind::Eof => break,
                _ => {}
            }
        }

        let src_ptr = src.as_ptr() as usize;
        let src_end = src_ptr + src.len();
        let ident_ptr = ident_ptr.expect("identifier present") as usize;
        let string_ptr = string_ptr.expect("string present") as usize;
        assert!((src_ptr..src_end).contains(&ident_ptr));
        assert!((src_ptr..src_end).contains(&string_ptr));
    }

    #[test]
    fn comments_are_skipped_and_produce_no_tokens() {
        // Top-level comment, inline comment, comment-only indented line.
        let src = "# top comment\ndef foo():\n    # inner comment\n    let x: int = 1\n";
        let kinds = collect_kinds(src);
        // No token should carry a '#' — comments must vanish entirely.
        for k in &kinds {
            if let TokenKind::Identifier(s) = k {
                assert!(!s.starts_with('#'), "comment leaked into identifier: {s}");
            }
        }
        // The function body must still be present.
        assert!(kinds.contains(&TokenKind::Def));
        assert!(kinds.contains(&TokenKind::Let));
        assert!(kinds.contains(&TokenKind::Indent));
    }

    #[test]
    fn invalid_dedent_is_reported() {
        let src = "def x():\n    let a = 1\n  let b = 2\n";
        let mut lexer = Lexer::new(src);
        let mut saw_error = false;
        while let Some(item) = lexer.next() {
            match item {
                Ok(tok) if matches!(tok.kind, TokenKind::Eof) => break,
                Ok(_) => {}
                Err(err) => {
                    saw_error = true;
                    assert_eq!(err.kind, LexErrorKind::InvalidDedent);
                    break;
                }
            }
        }
        assert!(saw_error);
    }
}
