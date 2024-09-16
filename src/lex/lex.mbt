// An O(N) lexer
// TODO: check int, float out of bound parsing
let reserved_keywords : @immut/hashmap.T[String, Token] = @immut/hashmap.of(
  [
    ("true", BOOL(true)),
    ("false", BOOL(false)),
    ("Unit", UNIT_T),
    ("Bool", BOOL_T),
    ("Int", INT_T),
    ("Double", DOUBLE_T),
    ("Array", ARRAY),
    ("not", NOT), // only non-keyword in moonbit here
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
  AfterFloatingDot
}

pub fn lex_string(input : String) -> Array[Lexeme]!LexError {
  let result : Array[Lexeme] = Array::new()
  // put a new line here manually so we always spit the last token
  let chars = (input + "\n").iter().collect()
  let index_ub = chars.length()
  let mut loc = new_loc()
  let mut last_loc = loc
  let mut mode = Default
  let mut number_acc = 0
  let mut number_acc_f64 = 0.0
  let mut number_divdened_f64 = 1.0
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
            Some(tok) => result.push((tok, loc, loc))
            None => raise LexError(UnexpectedChar(char, loc))
          }
        }
      }
      (Number, '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9') =>
        number_acc = number_acc * 10 + char.to_int() - '0'.to_int()
      (Number, '.') => {
        mode = AfterFloatingDot
        number_acc_f64 = number_acc.to_double()
        number_acc = 0
      }
      (Number, _) => {
        mode = Default
        result.push((I32(number_acc), last_loc, loc))
        number_acc = 0
        continue // don't consume
      }
      (
        AfterFloatingDot,
        '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9',
      ) => {
        number_acc_f64 = number_acc_f64 * 10 +
          (char.to_int() - '0'.to_int()).to_double()
        number_divdened_f64 *= 10
      }
      (AfterFloatingDot, _) => {
        mode = Default
        result.push((F64(number_acc_f64 / number_divdened_f64), last_loc, loc))
        number_acc_f64 = 0
        number_divdened_f64 = 1
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
          result.push((to_push, last_loc, loc))
          ident_acc.reset()
          continue // don't consume
        }
      }
      (AfterAssign, '=') => {
        mode = Default
        result.push((EQ, last_loc, loc))
        last_loc = loc
      }
      (AfterAssign, _) => {
        mode = Default
        last_loc = loc
        result.push((ASSIGN, last_loc, loc))
        continue // don' consume
      }
      (AfterDiv, '/') => mode = Comment
      (AfterDiv, _) => {
        mode = Default
        result.push((DIV, last_loc, loc))
        continue // don' consume
      }
      (AfterLess, '=') => {
        mode = Default
        result.push((LE, last_loc, loc))
      }
      (AfterLess, c) => raise LexError(UnexpectedChar(c, loc))
      (AfterMinus, '>') => {
        mode = Default
        result.push((ARROW, last_loc, loc))
      }
      (AfterMinus, _) => {
        mode = Default
        result.push((SUB, last_loc, loc))
        continue
      }
    }
    loc = loc.advance(char)
  }
  result.push((EOF, loc, loc))
  result
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
      #| 1.0007 9.
      #| 2147483647
      ,
    ).map(fn { _1 => _1.0 }),
    content=
      #|[FN, ID("make_adder"), LPAREN, ID("x"), COLON, INT_T, RPAREN, ARROW, LPAREN, INT_T, RPAREN, ARROW, INT_T, LBRACE, FN, ID("adder"), LPAREN, ID("y"), COLON, INT_T, RPAREN, ARROW, INT_T, LBRACE, ID("x"), ADD, ID("y"), RBRACE, SEMICOL, ID("adder"), RBRACE, SEMICOL, FN, ID("main"), LBRACE, ID("print_int"), LPAREN, LPAREN, ID("make_adder"), LPAREN, I32(3), RPAREN, RPAREN, LPAREN, I32(7), RPAREN, RPAREN, RBRACE, SEMICOL, F64(1.0007), F64(9), I32(2147483647), EOF]
    ,
  )
}
