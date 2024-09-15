test "array is mutable" {
  let a = [1, 2, 3]
  (fn(arr : Array[Int]) {
    arr[0] = 3
    ignore(arr.pop())
  })(a)
  assert_eq!(a, [3, 2])
}

test "deque is mutable" {
  let q : @deque.T[Int] = @deque.of([1, 2, 3])
  (fn(q : @deque.T[Int]) { ignore(q.pop_back()) })(q)
  assert_eq!(q.iter().collect(), [1, 2])
}

fn fst[U, V, W](pair : (U, V, W)) -> U {
  match pair {
    (a, _, _) => a
  }
}

test "parse a simple expression" {
  let input =
    #|x + y
  let tokens = @lex.lex_string!(input)
  inspect!(
    tokens.map(fst),
    content=
      #|[ID("x"), ADD, ID("y")]
    ,
  )
  let (expr, _) = expression(tokens[:])
  inspect!(
    expr,
    content=
      #|(x+y)
    ,
  )
}

test "parse a statement" {
  let tokens = @lex.lex_string!(
    #|  fn adder(y: Int) -> Int {
    #|    x + y
    #|  };
    #|  adder
    ,
  )
  inspect!(
    tokens.map(fst),
    content=
      #|[FN, ID("adder"), LPAREN, ID("y"), COLON, INT_T, RPAREN, ARROW, INT_T, LBRACE, ID("x"), ADD, ID("y"), RBRACE, SEMICOL, ID("adder")]
    ,
  )
  let (statement, _) = stmt(tokens[:])
  inspect!(
    statement,
    content=
      #|fn adder(y: Int) -> Int{(x+y)}; adder
    ,
  )
}

test "parse a program" {
  let tokens = @lex.lex_string!(
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
  )
  let program = program(tokens[:])
  inspect!(
    program,
    content=
      #|[fn make_adder((x, Int)) -> (Int) -> Int {fn adder(y: Int) -> Int{(x+y)}; adder}, fn main {print_int(make_adder(3)(7))}]
    ,
  )
}
