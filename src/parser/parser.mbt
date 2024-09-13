// According to https://taolun.moonbitlang.com/t/topic/882/2, we don't have GADT, so I have to hand roll like below
//
//type Program[Ann] Array[Top[Ann]]
//
//enum TopNaked[Ann] {
//  LetDecl(ID[Ann], Type[Ann], Exp[Ann])
//  FnDecl(ID[Ann], Array[(ID[Ann], Type[Ann])], Type[Ann], Exp[Ann])
//  TopErr(Array[@lex.Token])
//}
//
//type Top[Ann] (TopNaked[Ann], Ann)
//
//type ID[Ann] (String, Ann)
//
//enum StmtNaked[Ann] {
//  LetTupleIn(ID[Ann], Array[ID[Ann]], Type[Ann]?, Exp[Ann], Stmt[Ann])
//  LetIn(ID[Ann], Type[Ann]?, Exp[Ann], Stmt[Ann])
//  AssignIn(Exp[Ann], Exp[Ann], Stmt[Ann])
//  Expr(Exp[Ann])
//  StmtErr
//}
//
//type Stmt[Ann] (StmtNaked[Ann], Ann)
//
//enum Value[Ann] {
//  I32(Int)
//  F64(Double)
//  Bool(Bool)
//  Lambda(Array[(ID[Ann], Type[Ann]?)], Type[Ann]?, Exp[Ann])
//  Unit
//  ValueErr
//}
//
//enum Unop {
//  Not
//  Neg
//}
//
//enum Binop {
//  Eq
//  Le
//  Add
//  Sub
//  Mul
//  Div
//  Index
//}
//
//enum ExpNaked[Ann] {
//  Value(Value[Ann])
//  Tuple(Array[Exp[Ann]])
//  Block(Array[Exp[Ann]])
//  Unary(Unop, Exp[Ann])
//  Binary(Binop, Exp[Ann], Exp[Ann])
//  Apply(Exp[Ann], Exp[Ann], Array[Exp[Ann]]) // we need at least 2 expression to perform an apply
//  If(Exp[Ann], Exp[Ann], Exp[Ann]?)
//  Var(ID[Ann])
//  MakeArray(Exp[Ann], Exp[Ann])
//  ExpErr
//}
//
//type Exp[Ann] (ExpNaked[Ann], Ann)
//
//enum TypeNaked[Ann] {
//  Unit
//  Bool
//  Int
//  Double
//  Array(Type[Ann])
//  Tuple(Array[Type[Ann]])
//  Function(Array[Type[Ann]], Type[Ann])
//  TypeErr
//}
//
//type Type[Ann] (TypeNaked[Ann], Ann)
//
//struct ParseAnn {
//  src_loc : @lex.Loc
//}
//
//pub fn program(view : ArrayView[(@lex.Token, @lex.Loc)]) -> Program[ParseAnn] {
//  let program : Program[ParseAnn] = []
//  let mut view = view
//  while true {
//    match view {
//      [(@lex.FN, _), ..] => {
//        let (top_fn, rest) = top_fn(view)
//        program._.push(top_fn)
//        view = rest
//      }
//      [(@lex.LET, _), ..] => {
//        let (top_let, rest) = top_let(view)
//        program._.push(top_let)
//        view = rest
//      }
//      [(_, loc_start), ..] => {
//        let acc : Array[@lex.Token] = []
//        while true {
//          match view {
//            [(@lex.FN | @lex.LET, _), ..] => break
//            [(tok, _), .. as view_rest] => {
//              acc.push(tok)
//              view = view_rest
//            }
//          }
//        }
//        program._.push((TopErr(acc), { src_loc: loc_start }))
//      }
//    }
//  }
//  program
//}
//
//fn top_fn(
//  view : ArrayView[(@lex.Token, @lex.Loc)]
//) -> (Top[ParseAnn], ArrayView[(@lex.Token, @lex.Loc)]) {
//  match view {
//    [(@lex.FN, _), (@lex.ID(id), _), .. as view1] => {
//      let (params, rest1) = params_typed(view1)
//      match rest1 {
//        [(@lex.ARROW, _), .. as view2] => {
//          let (return_type, rest2) = type(view2)
//          let (fn_body, rest3) = fn_body(rest2)
//          FnDecl(id, params, return_type, fn_body)
//        }
//      }
//    }
//  }
//}
//
//fn top_let(
//  tokens : ArrayView[(@lex.Token, @lex.Loc)]
//) -> (Top[ParseAnn], ArrayView[(@lex.Token, @lex.Loc)]) {
//
//}
//
//fn params_typed(
//  tokens : ArrayView[(@lex.Token, @lex.Loc)]
//) -> (Array[(ID[ParseAnn], Type[ParseAnn])], ArrayView[(@lex.Token, @lex.Loc)]) {
//
//}
//
//test "array is mutable" {
//  let a = [1, 2, 3]
//  (fn(arr : Array[Int]) {
//    arr[0] = 3
//    ignore(arr.pop())
//  })(a)
//  assert_eq!(a, [3, 2])
//}
