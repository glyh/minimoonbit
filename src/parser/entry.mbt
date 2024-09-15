//fn main {
//  try {
//    let tokens = @lex.lex_string!(
//      #|  fn adder(y: Int) -> Int {
//      #|    x + y
//      #|  };
//      #|  adder
//      ,
//    )
//    inspect!(
//      tokens.map(fst),
//      content=
//        #|[FN, ID("adder"), LPAREN, ID("y"), COLON, INT_T, RPAREN, ARROW, INT_T, LBRACE, ID("x"), ADD, ID("y"), RBRACE, SEMICOL, ID("adder")]
//      ,
//    )
//    let program = stmt(tokens[:])
//    inspect!(
//      program,
//      content=
//        #|
//      ,
//    )
//  } catch {
//    e => println(e)
//  }
//}
