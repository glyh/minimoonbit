pub struct Loc {
  row : Int
  col : Int
  index : Int
} derive(Show)

pub enum LexError_ {
  UnexpectedChar(Char, Loc)
} derive(Show)

pub type! LexError LexError_

fn new_loc() -> Loc {
  { row: 1, col: 1, index: 0 }
}

fn advance(self : Loc, c : Char) -> Loc {
  if c == '\n' {
    self.advance_line()
  } else {
    { ..self, col: self.col + 1, index: self.index + 1 }
  }
}

fn advance_line(self : Loc) -> Loc {
  { row: self.row + 1, col: 1, index: self.index + 1 }
}

pub enum Token {
  ID(String)
  COLON
  ASSIGN
  SEMICOL
  FN
  LPAREN
  RPAREN
  ARROW
  COMMA
  LBRACE
  RBRACE
  EQ
  LE
  ADD
  SUB
  MUL
  DIV
  IF
  ELSE
  LBKT
  RBKT
  BOOL(Bool)
  ARRAY
  UNIT_T
  BOOL_T
  DOUBLE_T
  I32(Int)
  F64(Double)
  NOT
  INT_T
  LET
  EOF
} derive(Show, Eq)

typealias Lexeme = (Token, Loc, Loc)
