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
    { row: self.row + 1, col: 1, index: self.index + 1 }
  } else {
    { ..self, col: self.col + 1, index: self.index + 1 }
  }
}

pub enum Token {
  BOOL(Bool)
  UNIT_T
  BOOL_T
  INT_T
  DOUBLE_T
  ARRAY
  NOT
  IF
  ELSE
  FN
  LET
  I32(Int)
  F64(Double)
  ID(String)
  ADD
  SUB
  MUL
  DIV
  ASSIGN
  EQ
  LE
  LPAREN
  RPAREN
  LBKT
  RBKT
  LBRACE
  RBRACE
  ARROW
  COLON
  SEMICOL
  COMMA
  EOF
} derive(Show, Eq)

typealias Lexeme = (Token, Loc, Loc)
