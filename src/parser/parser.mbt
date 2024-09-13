According to https://taolun.moonbitlang.com/t/topic/882/2, we don't have GADT, so I have to hand roll like below

struct Ann {
  src_loc : @lex.Loc
}

type Program Array[Top]

enum TopNaked {
  LetDecl(ID, Type, Exp)
  FnDecl(ID, Array[(ID, Type)], Type, Exp)
  TopErr(Array[@lex.Token])
}

type Top (TopNaked, Ann)

type ID (String, Ann)

enum StmtNaked {
  LetTupleIn(ID, Array[ID], Type?, Exp, Stmt)
  LetIn(ID, Type?, Exp, Stmt)
  AssignIn(Exp, Exp, Stmt)
  Expr(Exp)
  StmtErr
}

type Stmt (StmtNaked, Ann)

enum Value {
  I32(Int)
  F64(Double)
  Bool(Bool)
  Lambda(Array[(ID, Type?)], Type?, Exp)
  Unit
  ValueErr
}

enum Unop {
  Not
  Neg
}

enum Binop {
  Eq
  Le
  Add
  Sub
  Mul
  Div
  Index
}

enum ExpNaked {
  Value(Value)
  Tuple(Array[Exp])
  Block(Array[Exp])
  Unary(Unop, Exp)
  Binary(Binop, Exp, Exp)
  Apply(Exp, Exp, Array[Exp]) // we need at least 2 expression to perform an apply
  If(Exp, Exp, Exp?)
  Var(ID)
  MakeArray(Exp, Exp)
  ExpErr
}

type Exp (ExpNaked, Ann)

enum TypeNaked {
  Unit
  Bool
  Int
  Double
  Array(Type)
  Tuple(Array[Type])
  Function(Array[Type], Type)
  TypeErr
}

type Type (TypeNaked, Ann)

pub fn program(view : ArrayView[(@lex.Token, @lex.Loc)]) -> Program {
  let program : Program = []
  let mut view = view
  while true {
    match view {
      [(@lex.FN, _), ..] => {
        let (top_fn, rest) = top_fn(view)
        program._.push(top_fn)
        view = rest
      }
      [(@lex.LET, _), ..] => {
        let (top_let, rest) = top_let(view)
        program._.push(top_let)
        view = rest
      }
      [(_, loc_start), ..] => {
        let acc : Array[@lex.Token] = []
        while true {
          match view {
            [(@lex.FN | @lex.LET, _), ..] => break
            [(tok, _), .. as view_rest] => {
              acc.push(tok)
              view = view_rest
            }
          }
        }
        program._.push((TopErr(acc), { src_loc: loc_start }))
      }
    }
  }
  program
}

fn top_fn(
  view : ArrayView[(@lex.Token, @lex.Loc)]
) -> (Top, ArrayView[(@lex.Token, @lex.Loc)]) {
  match view {
    [(@lex.FN, _), (@lex.ID(id), _), .. as view1] => {
      let (params, rest1) = params_typed(view1)
      match rest1 {
        [(@lex.ARROW, _), .. as view2] => {
          let (return_type, rest2) = type(view2)
          let (fn_body, rest3) = fn_body(rest2)
          FnDecl(id, params, return_type, fn_body)
        }
      }
    }
  }
}

fn top_let(
  tokens : ArrayView[(@lex.Token, @lex.Loc)]
) -> (Top, ArrayView[(@lex.Token, @lex.Loc)]) {

}

fn params_typed(
  tokens : ArrayView[(@lex.Token, @lex.Loc)]
) -> (Array[(ID, Type)], ArrayView[(@lex.Token, @lex.Loc)]) {

}

test "array is mutable" {
  let a = [1, 2, 3]
  (fn(arr : Array[Int]) {
    arr[0] = 3
    ignore(arr.pop())
  })(a)
  assert_eq!(a, [3, 2])
}
