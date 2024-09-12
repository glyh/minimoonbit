pub struct Loc {
  row : Int
  col : Int
  index : Int
} derive(Show)

pub type! ExpectLE

pub type! UnexpectedChar Char

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
  Let
  ID(String)
  COLON
  ASSIGN
  SEMICOL
  FN
  MAIN
  INIT
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
  DOT
  ARRAY
  MAKE
  UNIT_T
  BOOL_T
  DOUBLE_T
  INT(Int)
  NOT
  INT_T
  LET
} derive(Show)

let reserved_keywords : @immut/hashmap.T[String, Token] = @immut/hashmap.of(
  [
    ("true", BOOL(true)),
    ("false", BOOL(false)),
    ("Unit", UNIT_T),
    ("Bool", BOOL_T),
    ("Int", INT_T),
    ("Double", DOUBLE_T),
    ("Array", ARRAY),
    ("not", NOT),
    ("if", IF),
    ("else", ELSE),
    ("fn", FN),
    ("let", LET),
  ],
)

let symbols : @immut/hashmap.T[Char, Token] = @immut/hashmap.of(
  [
    ('.', DOT),
    ('+', ADD),
    ('*', MUL),
    ('/', DIV),
    ('(', LPAREN),
    (')', RPAREN),
    ('[', LBKT),
    (']', RBKT),
    ('{', LBRACE),
    ('}', RBRACE),
    (':', COLON),
    (';', SEMICOL),
    (',', COMMA),
  ],
)

enum LexState { // DFA State
  Default
  Comment
  Number
  Ident
  AfterAssign
  AfterDiv
  AfterLess
  AfterMinus
}

pub fn lex_string!(input : String) -> Array[(Token, Loc)] {
  let result : Array[(Token, Loc)] = Array::new()
  // put a new line here manually so we always spit the last token
  let chars = (input + "\n").iter().collect()
  let index_ub = chars.length()
  let mut loc = new_loc()
  let mut last_loc = loc
  let mut mode = Default
  let mut number_acc = 0
  let ident_acc = Buffer::new(size_hint=128)
  while loc.index < index_ub {
    let char = chars[loc.index]
    match (mode, char) {
      (Comment, '\r' | '\n') => {
        mode = Default
        last_loc = loc
      }
      (Comment, _) => ()
      (Default, ' ' | '\t' | '\r' | '\n') => ()
      (Default, char) => {
        let ascii = char.to_int()
        if '0'.to_int() <= ascii && ascii <= '9'.to_int() {
          mode = Number
          last_loc = loc
          number_acc = ascii - '0'.to_int()
        } else if ('a'.to_int() <= ascii && ascii <= 'z'.to_int()) ||
          ('A'.to_int() <= ascii && ascii <= 'Z'.to_int()) ||
          char == '_' {
          mode = Ident
          last_loc = loc
          ident_acc.write_char(char)
        } else if char == '=' {
          mode = AfterAssign
          last_loc = loc
        } else if char == '/' {
          mode = AfterDiv
          last_loc = loc
        } else if char == '<' {
          mode = AfterLess
          last_loc = loc
        } else if char == '-' {
          mode = AfterMinus
          last_loc = loc
        } else {
          match symbols.find(char) {
            Some(tok) => result.push((tok, loc))
            None => raise UnexpectedChar(char)
          }
        }
      }
      // operators
      (Number, '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') =>
        number_acc = number_acc * 10 + char.to_int() - '0'.to_int()
      (Number, _) => {
        mode = Default
        result.push((INT(number_acc), last_loc))
        number_acc = 0
        continue // don't consume
      }
      (Ident, _) => {
        let ascii = char.to_int()
        if ('a'.to_int() <= ascii && ascii <= 'z'.to_int()) ||
          ('A'.to_int() <= ascii && ascii <= 'Z'.to_int()) ||
          char == '_' ||
          ('0'.to_int() <= ascii && ascii <= '9'.to_int()) {
          ident_acc.write_char(char)
        } else {
          let ident_or_keyword = ident_acc.to_string()
          mode = Default
          let to_push = match reserved_keywords.find(ident_or_keyword) {
            Some(tok) => tok
            None => ID(ident_or_keyword)
          }
          result.push((to_push, last_loc))
          ident_acc.reset()
          continue // don't consume
        }
      }
      (AfterAssign, '=') => {
        mode = Default
        result.push((EQ, last_loc))
        last_loc = loc
      }
      (AfterAssign, _) => {
        mode = Default
        last_loc = loc
        result.push((ASSIGN, last_loc))
        continue // don' consume
      }
      (AfterDiv, '/') => mode = Comment
      (AfterDiv, _) => {
        mode = Default
        result.push((DIV, last_loc))
        continue // don' consume
      }
      (AfterLess, '=') => {
        mode = Default
        result.push((LE, last_loc))
      }
      (AfterLess, _) => raise ExpectLE
      (AfterMinus, '>') => {
        mode = Default
        result.push((ARROW, last_loc))
      }
      (AfterMinus, _) => {
        mode = Default
        result.push((SUB, last_loc))
        continue
      }
    }
    loc = loc.advance(char)
  }
  result
}

fn fst[U, V](pair : (U, V)) -> U {
  match pair {
    (a, _) => a
  }
}

test "simple lexing" {
  inspect!(
    lex_string!(
      #|fn make_adder(x: Int) -> (Int) -> Int {
      #|  fn adder(y: Int) -> Int {
      #|    x + y
      #|  };
      #|  adder
      #|};
      #|
      #|fn main {
      #|  print_int((make_adder(3))(7))
      #|};
      ,
    ).map(fst),
    content=
      #|[FN, ID("make_adder"), LPAREN, ID("x"), COLON, INT_T, RPAREN, ARROW, LPAREN, INT_T, RPAREN, ARROW, INT_T, LBRACE, FN, ID("adder"), LPAREN, ID("y"), COLON, INT_T, RPAREN, ARROW, INT_T, LBRACE, ID("x"), ADD, ID("y"), RBRACE, SEMICOL, ID("adder"), RBRACE, SEMICOL, FN, ID("main"), LBRACE, ID("print_int"), LPAREN, LPAREN, ID("make_adder"), LPAREN, INT(3), RPAREN, RPAREN, LPAREN, INT(7), RPAREN, RPAREN, RBRACE, SEMICOL]
    ,
  )
}
