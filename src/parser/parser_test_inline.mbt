test "parse simple program" {
  let program =
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
  let tokens = @lex.lex_string!(program)
  inspect!(
    tokens.map(fn { _1 => _1.0 }),
    content=
      #|[FN, ID("make_adder"), LPAREN, ID("x"), COLON, INT_T, RPAREN, ARROW, LPAREN, INT_T, RPAREN, ARROW, INT_T, LBRACE, FN, ID("adder"), LPAREN, ID("y"), COLON, INT_T, RPAREN, ARROW, INT_T, LBRACE, ID("x"), ADD, ID("y"), RBRACE, SEMICOL, ID("adder"), RBRACE, SEMICOL, FN, ID("main"), LBRACE, ID("print_int"), LPAREN, LPAREN, ID("make_adder"), LPAREN, I32(3), RPAREN, RPAREN, LPAREN, I32(7), RPAREN, RPAREN, RBRACE, SEMICOL, EOF]
    ,
  )
  let (ast, _) = parser().parse(tokens[:])
  inspect!(
    ast,
    content=
      #|Some(LetRec({name: ("make_adder", Fun([Int], Int)), args: [("x", Int)], body: LetRec({name: ("adder", Int), args: [("y", Int)], body: Prim(Var("x"), Var("y"), Add, kind=None)}, Var("adder"))}, LetRec({name: ("main", Unit), args: [], body: App(Var("print_int"), [App(App(Var("make_adder"), [Int(3)]), [Int(7)])])}, Unit)))
    ,
  )
}

test "parse inprod-loop.mbt" {
  let src =
    #|fn inprod(v1: Array[Double], v2: Array[Double], acc: Double, i: Int) -> Double {
    #|  if 0 <= i {
    #|    inprod(v1, v2, acc + v1[i] * v2[i], i - 1)
    #|  } else {
    #|    acc
    #|  }
    #|};
    #|
    #|fn main {
    #|  let v1 = Array::make(3, 1.23);
    #|  let v2 = Array::make(3, 4.56);
    #|  print_int(truncate(1000000.0 * inprod(v1, v2, 0.0, 2)))
    #|};
  inspect!(
    parse_program(src),
    content=
      #|LetRec({name: ("inprod", Double), args: [("v1", Array(Double)), ("v2", Array(Double)), ("acc", Double), ("i", Int)], body: If(LE(Int(0), Var("i")), App(Var("inprod"), [Var("v1"), Var("v2"), Prim(Var("acc"), Prim(Get(Var("v1"), Var("i")), Get(Var("v2"), Var("i")), Mul, kind=None), Add, kind=None), Prim(Var("i"), Int(1), Sub, kind=None)]), Var("acc"))}, LetRec({name: ("main", Unit), args: [], body: Let(("v1", Var({val: None})), Array(Int(3), Double(1.23)), Let(("v2", Var({val: None})), Array(Int(3), Double(4.56)), App(Var("print_int"), [App(Var("truncate"), [Prim(Double(1000000), App(Var("inprod"), [Var("v1"), Var("v2"), Double(0), Int(2)]), Mul, kind=None)])])))}, Unit))
    ,
  )
}

test "parse matmul.mbt" {
  let src =
    #|fn matmul(l: Int, m: Int, n: Int, a: Array[Array[Double]], b: Array[Array[Double]], c: Array[Array[Double]]) -> Unit {
    #|  fn loop1(i: Int) -> Unit {
    #|    if 0 <= i {
    #|      fn loop2(j: Int) -> Unit {
    #|        if 0 <= j {
    #|          fn loop3(k: Int) -> Unit {
    #|            if 0 <= k {
    #|              c[i][j] = c[i][j] + a[i][k] * b[k][j];
    #|              loop3(k - 1)
    #|            } else {
    #|              ()
    #|            }
    #|          };
    #|          let _ = loop3(m - 1);
    #|          loop2(j - 1)
    #|        } else {
    #|          ()
    #|        }
    #|      };
    #|      let _ = loop2(n - 1);
    #|      loop1(i - 1)
    #|    } else {
    #|      ()
    #|    }
    #|  };
    #|  loop1(l - 1)
    #|};
    #|
    #|fn main {
    #|  let dummy = Array::make(0, 0.0);
    #|  fn make(m: Int, n: Int) -> Array[Array[Double]] {
    #|    let mat = Array::make(m, dummy);
    #|    fn init(i: Int) -> Unit {
    #|      if 0 <= i {
    #|        mat[i] = Array::make(n, 0.0);
    #|        init(i - 1)
    #|      } else {
    #|        ()
    #|      }
    #|    };
    #|    let _ = init(m - 1);
    #|    mat
    #|  };
    #|  let a = make(2, 3);
    #|  let b = make(3, 2);
    #|  let c = make(2, 2);
    #|  a[0][0] = 1.0; a[0][1] = 2.0; a[0][2] = 3.0;
    #|  a[1][0] = 4.0; a[1][1] = 5.0; a[1][2] = 6.0;
    #|  b[0][0] = 7.0; b[0][1] = 8.0;
    #|  b[1][0] = 9.0; b[1][1] = 10.0;
    #|  b[2][0] = 11.0; b[2][1] = 12.0;
    #|  let _ = matmul(2,3,2,a,b,c);
    #|  let _ = print_int(truncate(c[0][0]));
    #|  let _ = print_endline();
    #|  let _ = print_int(truncate(c[0][1]));
    #|  let _ = print_endline();
    #|  let _ = print_int(truncate(c[1][0]));
    #|  let _ = print_endline();
    #|  let _ = print_int(truncate(c[1][1]));
    #|  ()
    #|};
  inspect!(
    parse_program(src),
    content=
      #|LetRec({name: ("matmul", Unit), args: [("l", Int), ("m", Int), ("n", Int), ("a", Array(Array(Double))), ("b", Array(Array(Double))), ("c", Array(Array(Double)))], body: LetRec({name: ("loop1", Unit), args: [("i", Int)], body: If(LE(Int(0), Var("i")), LetRec({name: ("loop2", Unit), args: [("j", Int)], body: If(LE(Int(0), Var("j")), LetRec({name: ("loop3", Unit), args: [("k", Int)], body: If(LE(Int(0), Var("k")), Put(Get(Get(Var("c"), Var("i")), Var("j")), Prim(Get(Get(Var("c"), Var("i")), Var("j")), Prim(Get(Get(Var("a"), Var("i")), Var("k")), Get(Get(Var("b"), Var("k")), Var("j")), Mul, kind=None), Add, kind=None), App(Var("loop3"), [Prim(Var("k"), Int(1), Sub, kind=None)])), Unit)}, Let(("_", Var({val: None})), App(Var("loop3"), [Prim(Var("m"), Int(1), Sub, kind=None)]), App(Var("loop2"), [Prim(Var("j"), Int(1), Sub, kind=None)]))), Unit)}, Let(("_", Var({val: None})), App(Var("loop2"), [Prim(Var("n"), Int(1), Sub, kind=None)]), App(Var("loop1"), [Prim(Var("i"), Int(1), Sub, kind=None)]))), Unit)}, App(Var("loop1"), [Prim(Var("l"), Int(1), Sub, kind=None)]))}, LetRec({name: ("main", Unit), args: [], body: Let(("dummy", Var({val: None})), Array(Int(0), Double(0)), LetRec({name: ("make", Array(Array(Double))), args: [("m", Int), ("n", Int)], body: Let(("mat", Var({val: None})), Array(Var("m"), Var("dummy")), LetRec({name: ("init", Unit), args: [("i", Int)], body: If(LE(Int(0), Var("i")), Put(Get(Var("mat"), Var("i")), Array(Var("n"), Double(0)), App(Var("init"), [Prim(Var("i"), Int(1), Sub, kind=None)])), Unit)}, Let(("_", Var({val: None})), App(Var("init"), [Prim(Var("m"), Int(1), Sub, kind=None)]), Var("mat"))))}, Let(("a", Var({val: None})), App(Var("make"), [Int(2), Int(3)]), Let(("b", Var({val: None})), App(Var("make"), [Int(3), Int(2)]), Let(("c", Var({val: None})), App(Var("make"), [Int(2), Int(2)]), Put(Get(Get(Var("a"), Int(0)), Int(0)), Double(1), Put(Get(Get(Var("a"), Int(0)), Int(1)), Double(2), Put(Get(Get(Var("a"), Int(0)), Int(2)), Double(3), Put(Get(Get(Var("a"), Int(1)), Int(0)), Double(4), Put(Get(Get(Var("a"), Int(1)), Int(1)), Double(5), Put(Get(Get(Var("a"), Int(1)), Int(2)), Double(6), Put(Get(Get(Var("b"), Int(0)), Int(0)), Double(7), Put(Get(Get(Var("b"), Int(0)), Int(1)), Double(8), Put(Get(Get(Var("b"), Int(1)), Int(0)), Double(9), Put(Get(Get(Var("b"), Int(1)), Int(1)), Double(10), Put(Get(Get(Var("b"), Int(2)), Int(0)), Double(11), Put(Get(Get(Var("b"), Int(2)), Int(1)), Double(12), Let(("_", Var({val: None})), App(Var("matmul"), [Int(2), Int(3), Int(2), Var("a"), Var("b"), Var("c")]), Let(("_", Var({val: None})), App(Var("print_int"), [App(Var("truncate"), [Get(Get(Var("c"), Int(0)), Int(0))])]), Let(("_", Var({val: None})), App(Var("print_endline"), []), Let(("_", Var({val: None})), App(Var("print_int"), [App(Var("truncate"), [Get(Get(Var("c"), Int(0)), Int(1))])]), Let(("_", Var({val: None})), App(Var("print_endline"), []), Let(("_", Var({val: None})), App(Var("print_int"), [App(Var("truncate"), [Get(Get(Var("c"), Int(1)), Int(0))])]), Let(("_", Var({val: None})), App(Var("print_endline"), []), Let(("_", Var({val: None})), App(Var("print_int"), [App(Var("truncate"), [Get(Get(Var("c"), Int(1)), Int(1))])]), Unit)))))))))))))))))))))))))}, Unit))
    ,
  )
}

test "nested fns" {
  let src =
    #|fn no_ann() -> Unit {
    #|  fn a(u: Int, v: Int, w) -> Int {
    #|    fn b(x: Bool, y, z: Bool) {
    #|      ()
    #|    }; ()
    #|  }; ()
    #|};
  inspect!(
    parse_program(src),
    content=
      #|LetRec({name: ("no_ann", Unit), args: [], body: LetRec({name: ("a", Int), args: [("u", Int), ("v", Int), ("w", Var({val: None}))], body: LetRec({name: ("b", Var({val: None})), args: [("x", Bool), ("y", Var({val: None})), ("z", Bool)], body: Unit}, Unit)}, Unit)}, Unit)
    ,
  )
}
