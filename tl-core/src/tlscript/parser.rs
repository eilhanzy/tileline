//! Recursive-descent parser for `.tlscript` (V1 skeleton).
//!
//! The parser consumes the zero-copy lexer stream and builds a borrowing-friendly AST. It uses
//! recursive-descent statement parsing plus precedence-climbing expression parsing.

use std::fmt;

use super::ast::*;
use super::lexer::{LexError, LexErrorKind};
use super::token::{Span, Token, TokenKind};

/// Parser error category.
#[derive(Debug, Clone, PartialEq)]
pub enum ParseErrorKind {
    /// Underlying lexer error forwarded by the parser.
    Lexer(LexErrorKind),
    /// Unexpected end of token stream.
    UnexpectedEof,
    /// Token did not match the expected grammar.
    UnexpectedToken { expected: &'static str },
    /// Expected an identifier token.
    ExpectedIdentifier,
    /// Expected a primitive type token.
    ExpectedTypeName,
    /// Expected an expression node.
    ExpectedExpression,
    /// Decorator placement or syntax is invalid for the current grammar.
    InvalidDecorator,
    /// Assignment target is not a supported lvalue in V1.
    InvalidAssignmentTarget,
    /// `for` loop does not use the supported `range(...)` form.
    InvalidForRange,
}

/// Parse error with source span.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    /// Error category.
    pub kind: ParseErrorKind,
    /// Error source span.
    pub span: Span,
}

impl ParseError {
    fn new(kind: ParseErrorKind, span: Span) -> Self {
        Self { kind, span }
    }
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} at line {}, column {}",
            self.kind, self.span.line, self.span.column
        )
    }
}

impl std::error::Error for ParseError {}

/// `.tlscript` parser over a token iterator.
pub struct Parser<'src, I>
where
    I: Iterator<Item = Result<Token<'src>, LexError>>,
{
    tokens: I,
    peeked: Option<Result<Token<'src>, LexError>>,
    last_span: Span,
}

impl<'src, I> Parser<'src, I>
where
    I: Iterator<Item = Result<Token<'src>, LexError>>,
{
    /// Construct a parser from a lexer/token iterator.
    pub fn new(tokens: I) -> Self {
        Self {
            tokens,
            peeked: None,
            last_span: Span::new(0, 0, 1, 1),
        }
    }

    /// Parse the full module.
    pub fn parse_module(&mut self) -> Result<Module<'src>, ParseError> {
        let mut items = Vec::new();
        self.skip_newlines()?;

        while !matches!(self.peek_kind()?, Some(TokenKind::Eof)) {
            let item = self.parse_item()?;
            items.push(item);
            self.skip_newlines()?;
        }

        let eof = self.expect_eof()?;
        let span = if let (Some(first), Some(last)) = (items.first(), items.last()) {
            join_spans(first.span(), last.span())
        } else {
            eof.span
        };

        Ok(Module { items, span })
    }

    fn parse_item(&mut self) -> Result<Item<'src>, ParseError> {
        let decorators = self.parse_decorators()?;
        let func = self.parse_function_def(decorators)?;
        Ok(Item::Function(func))
    }

    fn parse_decorators(&mut self) -> Result<Vec<Decorator<'src>>, ParseError> {
        let mut decorators = Vec::new();
        loop {
            let Some(tok) = self.peek_token()? else {
                return Err(self.error_here(ParseErrorKind::UnexpectedEof));
            };
            let decorator = match tok.kind {
                TokenKind::ExportDecorator => Decorator {
                    kind: DecoratorKind::Export,
                    span: tok.span,
                },
                TokenKind::Decorator(name) => Decorator {
                    kind: DecoratorKind::Named(name),
                    span: tok.span,
                },
                TokenKind::At => {
                    return Err(ParseError::new(ParseErrorKind::InvalidDecorator, tok.span));
                }
                _ => break,
            };
            self.next_token()?; // consume decorator token
            self.expect_newline()?;
            decorators.push(decorator);
            self.skip_newlines()?;
        }
        Ok(decorators)
    }

    fn parse_function_def(
        &mut self,
        decorators: Vec<Decorator<'src>>,
    ) -> Result<FunctionDef<'src>, ParseError> {
        let def_tok = self.expect_token(|k| matches!(k, TokenKind::Def), "def")?;
        let (name, name_span) = self.expect_identifier()?;
        self.expect_token(|k| matches!(k, TokenKind::LParen), "(")?;
        let params = self.parse_params()?;
        self.expect_token(|k| matches!(k, TokenKind::RParen), ")")?;

        let return_type = if matches!(self.peek_kind()?, Some(TokenKind::Arrow)) {
            self.next_token()?; // ->
            Some(self.parse_type_annotation()?)
        } else {
            None
        };

        self.expect_token(|k| matches!(k, TokenKind::Colon), ":")?;
        let body = self.parse_block()?;

        let start_span = decorators.first().map(|d| d.span).unwrap_or(def_tok.span);
        let span = join_spans(start_span, body.span);
        Ok(FunctionDef {
            decorators,
            name,
            name_span,
            params,
            return_type,
            body,
            span,
        })
    }

    fn parse_params(&mut self) -> Result<Vec<Param<'src>>, ParseError> {
        let mut params = Vec::new();
        if matches!(self.peek_kind()?, Some(TokenKind::RParen)) {
            return Ok(params);
        }

        loop {
            let (name, name_span) = self.expect_identifier()?;
            let ty = if matches!(self.peek_kind()?, Some(TokenKind::Colon)) {
                self.next_token()?;
                Some(self.parse_type_annotation()?)
            } else {
                None
            };
            let span = ty
                .map(|t| join_spans(name_span, t.span))
                .unwrap_or(name_span);
            params.push(Param {
                name,
                name_span,
                ty,
                span,
            });

            if matches!(self.peek_kind()?, Some(TokenKind::Comma)) {
                self.next_token()?;
                if matches!(self.peek_kind()?, Some(TokenKind::RParen)) {
                    break;
                }
                continue;
            }
            break;
        }
        Ok(params)
    }

    fn parse_block(&mut self) -> Result<Block<'src>, ParseError> {
        let nl = self.expect_newline()?;
        let indent = self.expect_token(|k| matches!(k, TokenKind::Indent), "Indent")?;

        let mut statements = Vec::new();
        self.skip_newlines()?;

        while !matches!(self.peek_kind()?, Some(TokenKind::Dedent | TokenKind::Eof)) {
            let stmt = self.parse_stmt()?;
            statements.push(stmt);
            self.skip_newlines()?;
        }

        let dedent = self.expect_token(|k| matches!(k, TokenKind::Dedent), "Dedent")?;
        let span = if let (Some(_first), Some(last)) = (statements.first(), statements.last()) {
            join_spans(indent.span, last.span()).max_start(nl.span.start)
        } else {
            join_spans(nl.span, dedent.span)
        };
        Ok(Block { statements, span })
    }

    fn parse_stmt(&mut self) -> Result<Stmt<'src>, ParseError> {
        match self.peek_kind()? {
            Some(TokenKind::Let) => self.parse_let_stmt().map(Stmt::Let),
            Some(TokenKind::If) => self.parse_if_stmt().map(Stmt::If),
            Some(TokenKind::While) => self.parse_while_stmt().map(Stmt::While),
            Some(TokenKind::For) => self.parse_for_range_stmt().map(Stmt::ForRange),
            Some(TokenKind::Dedent | TokenKind::Eof) => {
                Err(self.error_here(ParseErrorKind::UnexpectedToken {
                    expected: "statement",
                }))
            }
            _ => self.parse_assignment_or_expr_stmt(),
        }
    }

    fn parse_let_stmt(&mut self) -> Result<LetStmt<'src>, ParseError> {
        let let_tok = self.expect_token(|k| matches!(k, TokenKind::Let), "let")?;
        let (name, name_span) = self.expect_identifier()?;
        let ty = if matches!(self.peek_kind()?, Some(TokenKind::Colon)) {
            self.next_token()?;
            Some(self.parse_type_annotation()?)
        } else {
            None
        };
        self.expect_token(|k| matches!(k, TokenKind::Assign), "=")?;
        let value = self.parse_expr()?;
        let end_span = self.consume_statement_terminator_span()?;

        Ok(LetStmt {
            name,
            name_span,
            ty,
            value,
            span: join_spans(let_tok.span, end_span),
        })
    }

    fn parse_if_stmt(&mut self) -> Result<IfStmt<'src>, ParseError> {
        let if_tok = self.expect_token(|k| matches!(k, TokenKind::If), "if")?;
        let first_cond = self.parse_expr()?;
        self.expect_token(|k| matches!(k, TokenKind::Colon), ":")?;
        let first_body = self.parse_block()?;

        let mut branches = vec![IfBranch {
            span: join_spans(if_tok.span, first_body.span),
            condition: first_cond,
            body: first_body,
        }];

        while matches!(self.peek_kind()?, Some(TokenKind::Elif)) {
            let elif_tok = self.next_token()?;
            let cond = self.parse_expr()?;
            self.expect_token(|k| matches!(k, TokenKind::Colon), ":")?;
            let body = self.parse_block()?;
            let span = join_spans(elif_tok.span, body.span);
            branches.push(IfBranch {
                condition: cond,
                body,
                span,
            });
        }

        let else_block = if matches!(self.peek_kind()?, Some(TokenKind::Else)) {
            self.next_token()?; // else
            self.expect_token(|k| matches!(k, TokenKind::Colon), ":")?;
            Some(self.parse_block()?)
        } else {
            None
        };

        let end_span = else_block
            .as_ref()
            .map(|b| b.span)
            .unwrap_or_else(|| branches.last().expect("branches non-empty").span);
        Ok(IfStmt {
            branches,
            else_block,
            span: join_spans(if_tok.span, end_span),
        })
    }

    fn parse_while_stmt(&mut self) -> Result<WhileStmt<'src>, ParseError> {
        let while_tok = self.expect_token(|k| matches!(k, TokenKind::While), "while")?;
        let condition = self.parse_expr()?;
        self.expect_token(|k| matches!(k, TokenKind::Colon), ":")?;
        let body = self.parse_block()?;
        let span = join_spans(while_tok.span, body.span);
        Ok(WhileStmt {
            condition,
            body,
            span,
        })
    }

    fn parse_for_range_stmt(&mut self) -> Result<ForRangeStmt<'src>, ParseError> {
        let for_tok = self.expect_token(|k| matches!(k, TokenKind::For), "for")?;
        let (binding, binding_span) = self.expect_identifier()?;
        self.expect_token(|k| matches!(k, TokenKind::In), "in")?;
        let range = self.parse_range_spec()?;
        self.expect_token(|k| matches!(k, TokenKind::Colon), ":")?;
        let body = self.parse_block()?;
        let span = join_spans(for_tok.span, body.span);
        Ok(ForRangeStmt {
            binding,
            binding_span,
            range,
            body,
            span,
        })
    }

    fn parse_range_spec(&mut self) -> Result<RangeSpec<'src>, ParseError> {
        let (callee, callee_span) = self.expect_identifier()?;
        if callee != "range" {
            return Err(ParseError::new(
                ParseErrorKind::InvalidForRange,
                callee_span,
            ));
        }
        self.expect_token(|k| matches!(k, TokenKind::LParen), "(")?;
        let mut args = Vec::new();
        if !matches!(self.peek_kind()?, Some(TokenKind::RParen)) {
            loop {
                args.push(self.parse_expr()?);
                if matches!(self.peek_kind()?, Some(TokenKind::Comma)) {
                    self.next_token()?;
                    if matches!(self.peek_kind()?, Some(TokenKind::RParen)) {
                        break;
                    }
                    continue;
                }
                break;
            }
        }
        let rparen = self.expect_token(|k| matches!(k, TokenKind::RParen), ")")?;
        if args.is_empty() || args.len() > 3 {
            return Err(ParseError::new(
                ParseErrorKind::InvalidForRange,
                callee_span,
            ));
        }
        Ok(RangeSpec {
            callee_span,
            span: join_spans(callee_span, rparen.span),
            args,
        })
    }

    fn parse_assignment_or_expr_stmt(&mut self) -> Result<Stmt<'src>, ParseError> {
        let expr = self.parse_expr()?;
        if matches!(self.peek_kind()?, Some(TokenKind::Assign)) {
            let assign_tok = self.next_token()?;
            let value = self.parse_expr()?;
            let end_span = self.consume_statement_terminator_span()?;
            match expr.kind {
                ExprKind::Identifier(name) => Ok(Stmt::Assign(AssignStmt {
                    target: name,
                    target_span: expr.span,
                    value,
                    span: join_spans(expr.span, end_span.max_start(assign_tok.span.start)),
                })),
                _ => Err(ParseError::new(
                    ParseErrorKind::InvalidAssignmentTarget,
                    expr.span,
                )),
            }
        } else {
            let end_span = self.consume_statement_terminator_span()?;
            Ok(Stmt::Expr(ExprStmt {
                span: join_spans(expr.span, end_span),
                expr,
            }))
        }
    }

    fn parse_expr(&mut self) -> Result<Expr<'src>, ParseError> {
        self.parse_expr_prec(1)
    }

    fn parse_expr_prec(&mut self, min_prec: u8) -> Result<Expr<'src>, ParseError> {
        let mut left = self.parse_unary()?;

        loop {
            let Some((op, prec)) = self.peek_binary_op()? else {
                break;
            };
            if prec < min_prec {
                break;
            }
            let _op_tok = self.next_token()?; // consume operator
            let right = self.parse_expr_prec(prec + 1)?;
            let span = join_spans(left.span, right.span);
            left = Expr::new(
                ExprKind::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                },
                span,
            );
        }

        Ok(left)
    }

    fn parse_unary(&mut self) -> Result<Expr<'src>, ParseError> {
        if matches!(self.peek_kind()?, Some(TokenKind::Minus)) {
            let minus = self.next_token()?;
            let expr = self.parse_unary()?;
            let span = join_spans(minus.span, expr.span);
            return Ok(Expr::new(
                ExprKind::Unary {
                    op: UnaryOp::Neg,
                    expr: Box::new(expr),
                },
                span,
            ));
        }
        self.parse_postfix()
    }

    fn parse_postfix(&mut self) -> Result<Expr<'src>, ParseError> {
        let mut expr = self.parse_primary()?;
        loop {
            if !matches!(self.peek_kind()?, Some(TokenKind::LParen)) {
                break;
            }
            self.next_token()?; // (
            let mut args = Vec::new();
            if !matches!(self.peek_kind()?, Some(TokenKind::RParen)) {
                loop {
                    args.push(self.parse_expr()?);
                    if matches!(self.peek_kind()?, Some(TokenKind::Comma)) {
                        self.next_token()?;
                        if matches!(self.peek_kind()?, Some(TokenKind::RParen)) {
                            break;
                        }
                        continue;
                    }
                    break;
                }
            }
            let rparen = self.expect_token(|k| matches!(k, TokenKind::RParen), ")")?;
            let span = join_spans(expr.span, rparen.span);
            expr = Expr::new(
                ExprKind::Call {
                    callee: Box::new(expr),
                    args,
                },
                span,
            );
        }
        Ok(expr)
    }

    fn parse_primary(&mut self) -> Result<Expr<'src>, ParseError> {
        let tok = self.next_token()?;
        match tok.kind {
            TokenKind::Identifier(name) => Ok(Expr::new(ExprKind::Identifier(name), tok.span)),
            TokenKind::Integer(v) => Ok(Expr::new(ExprKind::IntegerLiteral(v), tok.span)),
            TokenKind::Float(v) => Ok(Expr::new(ExprKind::FloatLiteral(v), tok.span)),
            TokenKind::String(v) => Ok(Expr::new(ExprKind::StringLiteral(v), tok.span)),
            TokenKind::True => Ok(Expr::new(ExprKind::BoolLiteral(true), tok.span)),
            TokenKind::False => Ok(Expr::new(ExprKind::BoolLiteral(false), tok.span)),
            TokenKind::LParen => {
                let inner = self.parse_expr()?;
                let rparen = self.expect_token(|k| matches!(k, TokenKind::RParen), ")")?;
                let span = join_spans(tok.span, rparen.span);
                Ok(Expr::new(ExprKind::Grouping(Box::new(inner)), span))
            }
            _ => Err(ParseError::new(
                ParseErrorKind::ExpectedExpression,
                tok.span,
            )),
        }
    }

    fn parse_type_annotation(&mut self) -> Result<TypeAnnotation, ParseError> {
        let tok = self.next_token()?;
        let kind = match tok.kind {
            TokenKind::TypeInt => TypeName::Int,
            TokenKind::TypeFloat => TypeName::Float,
            TokenKind::TypeBool => TypeName::Bool,
            TokenKind::TypeStr => TypeName::Str,
            _ => return Err(ParseError::new(ParseErrorKind::ExpectedTypeName, tok.span)),
        };
        Ok(TypeAnnotation {
            kind,
            span: tok.span,
        })
    }

    fn consume_statement_terminator_span(&mut self) -> Result<Span, ParseError> {
        match self.peek_kind()? {
            Some(TokenKind::Newline) => Ok(self.next_token()?.span),
            Some(TokenKind::Dedent | TokenKind::Eof) => Ok(self.peek_token()?.expect_some().span),
            _ => Err(self.error_here(ParseErrorKind::UnexpectedToken {
                expected: "newline/dedent/eof",
            })),
        }
    }

    fn skip_newlines(&mut self) -> Result<(), ParseError> {
        while matches!(self.peek_kind()?, Some(TokenKind::Newline)) {
            self.next_token()?;
        }
        Ok(())
    }

    fn expect_newline(&mut self) -> Result<Token<'src>, ParseError> {
        self.expect_token(|k| matches!(k, TokenKind::Newline), "newline")
    }

    fn expect_identifier(&mut self) -> Result<(&'src str, Span), ParseError> {
        let tok = self.next_token()?;
        if let TokenKind::Identifier(name) = tok.kind {
            Ok((name, tok.span))
        } else {
            Err(ParseError::new(
                ParseErrorKind::ExpectedIdentifier,
                tok.span,
            ))
        }
    }

    fn expect_eof(&mut self) -> Result<Token<'src>, ParseError> {
        self.expect_token(|k| matches!(k, TokenKind::Eof), "eof")
    }

    fn expect_token<P>(
        &mut self,
        predicate: P,
        expected: &'static str,
    ) -> Result<Token<'src>, ParseError>
    where
        P: FnOnce(TokenKind<'src>) -> bool,
    {
        let tok = self.next_token()?;
        if predicate(tok.kind) {
            Ok(tok)
        } else {
            Err(ParseError::new(
                ParseErrorKind::UnexpectedToken { expected },
                tok.span,
            ))
        }
    }

    fn peek_binary_op(&mut self) -> Result<Option<(BinaryOp, u8)>, ParseError> {
        let op = match self.peek_kind()? {
            Some(TokenKind::OrOr) => (BinaryOp::OrOr, 1),
            Some(TokenKind::AndAnd) => (BinaryOp::AndAnd, 2),
            Some(TokenKind::EqEq) => (BinaryOp::EqEq, 3),
            Some(TokenKind::NotEq) => (BinaryOp::NotEq, 3),
            Some(TokenKind::Lt) => (BinaryOp::Lt, 4),
            Some(TokenKind::LtEq) => (BinaryOp::LtEq, 4),
            Some(TokenKind::Gt) => (BinaryOp::Gt, 4),
            Some(TokenKind::GtEq) => (BinaryOp::GtEq, 4),
            Some(TokenKind::Plus) => (BinaryOp::Add, 5),
            Some(TokenKind::Minus) => (BinaryOp::Sub, 5),
            Some(TokenKind::Star) => (BinaryOp::Mul, 6),
            Some(TokenKind::Slash) => (BinaryOp::Div, 6),
            Some(TokenKind::Percent) => (BinaryOp::Mod, 6),
            _ => return Ok(None),
        };
        Ok(Some(op))
    }

    fn peek_kind(&mut self) -> Result<Option<TokenKind<'src>>, ParseError> {
        Ok(self.peek_token()?.map(|t| t.kind))
    }

    fn peek_token(&mut self) -> Result<Option<Token<'src>>, ParseError> {
        if self.peeked.is_none() {
            self.peeked = self.tokens.next();
        }
        match self.peeked.as_ref() {
            Some(Ok(tok)) => Ok(Some(*tok)),
            Some(Err(err)) => Err(ParseError::new(ParseErrorKind::Lexer(err.kind), err.span)),
            None => Ok(None),
        }
    }

    fn next_token(&mut self) -> Result<Token<'src>, ParseError> {
        let next = if let Some(v) = self.peeked.take() {
            Some(v)
        } else {
            self.tokens.next()
        };

        match next {
            Some(Ok(tok)) => {
                self.last_span = tok.span;
                Ok(tok)
            }
            Some(Err(err)) => Err(ParseError::new(ParseErrorKind::Lexer(err.kind), err.span)),
            None => Err(self.error_here(ParseErrorKind::UnexpectedEof)),
        }
    }

    fn error_here(&self, kind: ParseErrorKind) -> ParseError {
        ParseError::new(kind, self.last_span)
    }
}

trait ExpectSomeToken<'src> {
    fn expect_some(self) -> Token<'src>;
}

impl<'src> ExpectSomeToken<'src> for Option<Token<'src>> {
    fn expect_some(self) -> Token<'src> {
        self.expect("parser internal invariant: token lookahead exists")
    }
}

fn join_spans(a: Span, b: Span) -> Span {
    let (line, column) = if a.start <= b.start {
        (a.line, a.column)
    } else {
        (b.line, b.column)
    };
    Span::new(a.start.min(b.start), a.end.max(b.end), line, column)
}

trait SpanExt {
    fn max_start(self, start: usize) -> Self;
}

impl SpanExt for Span {
    fn max_start(mut self, start: usize) -> Self {
        if self.start < start {
            self.start = start;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlscript::Lexer;

    fn parse_ok(src: &str) -> Module<'_> {
        let mut parser = Parser::new(Lexer::new(src));
        parser.parse_module().expect("parse ok")
    }

    #[test]
    fn parses_exported_function_with_if_else() {
        let module = parse_ok(concat!(
            "@export\n",
            "def update(dt: float) -> int:\n",
            "    let x: int = 1\n",
            "    if dt > 0.0:\n",
            "        x = x + 1\n",
            "    else:\n",
            "        x = x - 1\n",
        ));

        assert_eq!(module.items.len(), 1);
        let Item::Function(func) = &module.items[0];
        assert_eq!(func.name, "update");
        assert_eq!(func.decorators.len(), 1);
        assert!(matches!(func.decorators[0].kind, DecoratorKind::Export));
        assert!(matches!(
            func.return_type,
            Some(TypeAnnotation {
                kind: TypeName::Int,
                ..
            })
        ));
        assert!(!func.body.statements.is_empty());
    }

    #[test]
    fn parses_for_range_and_call_expression() {
        let module = parse_ok(concat!(
            "def sim():\n",
            "    for i in range(0, 10):\n",
            "        tick(i)\n",
        ));

        let Item::Function(func) = &module.items[0];
        match &func.body.statements[0] {
            Stmt::ForRange(for_stmt) => assert_eq!(for_stmt.range.args.len(), 2),
            other => panic!("expected for-range stmt, got {other:?}"),
        }
    }

    #[test]
    fn expression_precedence_is_respected() {
        let module = parse_ok("def f():\n    let x = 1 + 2 * 3\n");
        let Item::Function(func) = &module.items[0];
        let Stmt::Let(let_stmt) = &func.body.statements[0] else {
            panic!("expected let stmt");
        };
        match &let_stmt.value.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left: _,
                right,
            } => {
                assert!(matches!(
                    right.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
            }
            other => panic!("unexpected expr shape: {other:?}"),
        }
    }
}
