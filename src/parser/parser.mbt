fn tier_left(op : Parser[B], next : Parser[S]) -> Parser[S] {
  fold_left(
    next,
    next,
    op.map(
      fn(op) {
        fn(lhs, rhs) {
          S::Prim(lhs, rhs, op, kind=@types.classed_tvar([Arith]))
        }
      },
    ),
  )
}

fn fix_type(ty : T?) -> T {
  match ty {
    Some(ty) => ty
    None => @types.dummy_tvar()
  }
}

fn parser() -> PS {
  let expr_ref : Ref[PS] = { val: empty() }
  let stmt_ref : Ref[PS] = { val: empty() }
  let block_expr = tok(@lex.LBRACE) >> ref(stmt_ref) << tok(@lex.RBRACE)

  // Type
  let type_ref : Ref[PT] = { val: empty() }
  type_ref.val = tok(@lex.UNIT_T).replace(T::Unit) +
    tok(@lex.BOOL_T).replace(T::Bool) +
    tok(@lex.INT_T).replace(T::Int) +
    tok(@lex.DOUBLE_T).replace(T::Double) +
    (
      tok(@lex.ARRAY) >>
      tok(@lex.LBKT) >>
      ref(type_ref).map(T::Array) <<
      tok(@lex.RBKT)
    ) +
    lift2(
      fn(args, ret) { T::Fun(args, ret) },
      tok(@lex.LPAREN) >>
      sep_list(tok(@lex.COMMA), ref(type_ref)) <<
      tok(@lex.RPAREN),
      tok(@lex.ARROW) >> ref(type_ref),
    ) +
    (
      tok(@lex.LPAREN) >>
      sep_list1(tok(@lex.COMMA), ref(type_ref)).map(
        fn {
          [ty] => ty
          tys => T::Tuple(tys)
        },
      ) <<
      tok(@lex.RPAREN)
    )

  // Value
  let value_ref : Ref[PS] = { val: empty() }
  value_ref.val = (
      tok(@lex.LPAREN) >>
      sep_list(tok(@lex.COMMA), ref(expr_ref)).map(
        fn {
          [] => S::Unit
          [x] => x
          tuple => S::Tuple(tuple)
        },
      ) <<
      tok(@lex.RPAREN)
    ) +
    bool.map(S::Bool) +
    id.map(S::Var) +
    block_expr +
    (
      tok(@lex.SUB) >>
      ref(value_ref).map(
        fn { x => S::Neg(x, kind=@types.classed_tvar([Arith])) },
      )
    ) +
    double.map(S::Double) +
    int.map(S::Int) +
    (
      tok(@lex.NOT) >>
      tok(@lex.LPAREN) >>
      ref(expr_ref).map(S::Not) <<
      tok(@lex.RPAREN)
    ) +
    (
      tok(@lex.ARRAY) >>
      tok(@lex.COLON) >>
      tok(@lex.COLON) >>
      id_p(fn(s) { s == "make" }) >>
      tok(@lex.LPAREN) >>
      lift2(
        fn(e1, e2) { S::Array(e1, e2) },
        ref(expr_ref),
        tok(@lex.COMMA) >> ref(expr_ref),
      ) <<
      tok(@lex.RPAREN)
    )

  // Get or apply
  let get_or_apply_postfix : Parser[(S) -> S] = (
      tok(@lex.LBKT) >>
      ref(expr_ref).map(fn { accessor => fn { arr => S::Get(arr, accessor) } }) <<
      tok(@lex.RBKT)
    ) +
    (
      tok(@lex.LPAREN) >>
      sep_list(tok(@lex.COMMA), ref(expr_ref)).map(
        fn { params => fn { f => S::App(f, params) } },
      ) <<
      tok(@lex.RPAREN)
    )
  let get_or_apply : PS = fold_left(
    ref(value_ref),
    get_or_apply_postfix,
    epsilon().replace(fn(u, v) { v(u) }),
  )

  // If
  let if_ : PS = get_or_apply +
    lift3(
      fn(_test, _then, _else) { S::If(_test, _then, _else) },
      tok(@lex.IF) >> ref(expr_ref),
      block_expr,
      (tok(@lex.ELSE) >> block_expr) + epsilon().replace(S::Unit),
    )

  // Mul
  let mul_op = tok(@lex.MUL).replace(B::Mul) + tok(@lex.DIV).replace(B::Div)
  let mul_ : PS = tier_left(mul_op, if_)

  // Add
  let add_op = tok(@lex.ADD).replace(B::Add) + tok(@lex.SUB).replace(B::Sub)
  let add_ : PS = tier_left(add_op, mul_)

  // Cmp
  let cmp_op = tok(@lex.LE).replace(fn(lhs : S, rhs : S) { S::LE(lhs, rhs) }) +
    tok(@lex.EQ).replace(fn(lhs : S, rhs : S) { S::Eq(lhs, rhs) })
  let cmp_ : PS = lift3(fn(lhs, op, rhs) { op(lhs, rhs) }, add_, cmp_op, add_) +
    add_
  expr_ref.val = cmp_

  // Statement
  let type_annotation = tok(@lex.COLON) >> ref(type_ref)
  let type_result = tok(@lex.ARROW) >> ref(type_ref)
  fn let_tup_head(force_annotation : Bool) -> Parser[(S) -> S] {
    lift3_maybe(
      fn(ids : Array[String], ty : T?, rhs) -> ((S) -> S)? {
        let ids_typed = match ty {
          Some(T::Tuple(tys)) =>
            if tys.length() != ids.length() {
              return None
            } else {
              ids.mapi(fn(i, name) { (name, tys[i]) })
            }
          Some(_) => return None
          None => ids.map(fn { name => (name, @types.dummy_tvar()) })
        }
        Some(fn(rest) { S::LetTuple(ids_typed, rhs, rest) })
      },
      tok(@lex.LET) >>
      tok(@lex.LPAREN) >>
      sep_list1(tok(@lex.COMMA), id) <<
      tok(@lex.RPAREN),
      if force_annotation {
        type_annotation.map(Option::Some)
      } else {
        maybe(type_annotation)
      },
      tok(@lex.ASSIGN) >> ref(expr_ref),
    )
  }

  fn let_head(force_annotation : Bool) -> Parser[(S) -> S] {
    lift3(
      fn(id, ty, rhs) { fn(rest) { S::Let((id, fix_type(ty)), rhs, rest) } },
      tok(@lex.LET) >> id,
      if force_annotation {
        type_annotation.map(Option::Some)
      } else {
        maybe(type_annotation)
      },
      tok(@lex.ASSIGN) >> ref(expr_ref),
    )
  }

  fn fn_head(
    force_annotation : Bool,
    id_filter : (String) -> Bool
  ) -> Parser[(S) -> S] {
    let param = lift2(
      fn(id : String, ty : T?) { (id, fix_type(ty)) },
      id,
      if force_annotation {
        type_annotation.map(Option::Some)
      } else {
        maybe(type_annotation)
      },
    )
    lift4(
      fn(id : String, args : Array[(String, T)], ret_ty : T?, body : S) {
        fn(rest) {
          S::LetRec({ name: (id, fix_type(ret_ty)), args, body }, rest)
        }
      },
      tok(@lex.FN) >> id_p(id_filter),
      tok(@lex.LPAREN) >> sep_list(tok(@lex.COMMA), param) << tok(@lex.RPAREN),
      if force_annotation {
        type_result.map(Option::Some)
      } else {
        maybe(type_result)
      },
      block_expr,
    )
  }

  let stmt_head = fn_head(false, fn { _ => true }) +
    let_head(false) +
    let_tup_head(false) +
    lift2_maybe(
      fn(get_expr, rhs) {
        match get_expr {
          S::Get(_, _) => ()
          _ => return None
        }
        Some(fn(rest) { S::Put(get_expr, rhs, rest) })
      },
      get_or_apply,
      tok(@lex.ASSIGN) >> ref(expr_ref),
    )
  stmt_ref.val = fold_right(
    ref(expr_ref),
    stmt_head,
    tok(@lex.SEMICOL).replace(fn(u, v) { u(v) }),
  )
  let top_head = lift2(
      fn(id : String, body : S) {
        fn(rest) { S::LetRec({ name: (id, T::Unit), args: [], body }, rest) }
      },
      tok(@lex.FN) >> id_p(fn(s) { s == "main" || s == "init" }),
      block_expr,
    ) +
    fn_head(
      true,
      fn {
        "main" | "init" => false
        _ => true
      },
    ) +
    let_head(true)
  fold_right(
    tok(@lex.EOF).replace(S::Unit),
    // NOTE: top level definitions must start at the beginning of line
    line_start >> top_head,
    tok(@lex.SEMICOL).replace(fn(u, v) { u(v) }),
  )
}
