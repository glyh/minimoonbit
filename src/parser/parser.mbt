enum GetApplyPostFix {
  Index(S)
  Apply(Array[S])
}

fn merge_postfix(inner : S, outers : Array[GetApplyPostFix]) -> S {
  loop (inner, outers[:]) {
    (acc, []) => break acc
    (acc, [Index(accessor), .. as rest]) =>
      continue (S::Get(acc, accessor), rest)
    (acc, [Apply(params), .. as rest]) => continue (S::App(acc, params), rest)
  }
}

fn fix_type(ty : T?) -> T {
  match ty {
    Some(ty) => ty
    None => T::Var({ val: None })
  }
}

fn parser() -> PS {
  let expr_ref : Ref[PS] = { val: empty() }
  let stmt_ref : Ref[PS] = { val: empty() }
  let block_expr = tok(@lex.LBRACE) >> ref(stmt_ref) << tok(@lex.RBRACE)

  // Type
  let type_ref : Ref[PT] = { val: empty() }
  type_ref.val = tok(@lex.UNIT_T).map(fn { _ => T::Unit }) +
    tok(@lex.INT_T).map(fn { _ => T::Int }) +
    tok(@lex.DOUBLE_T).map(fn { _ => T::Double }) +
    (
      tok(@lex.ARRAY) >>
      tok(@lex.LBKT) >>
      ref(type_ref).map(T::Array) <<
      tok(@lex.RBKT)
    ) +
    lift2(
      fn(args, ret) { T::Fun(args, ret) },
      tok(@lex.LPAREN) >> sep_list1(tok(@lex.COMMA), ref(type_ref)),
      tok(@lex.RPAREN) >> tok(@lex.ARROW) >> ref(type_ref),
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
      tok(@lex.LPAREN) >> tok(@lex.RPAREN).map(fn { _ => S::Unit })
    ) +
    (
      tok(@lex.LPAREN) >>
      sep_list1(tok(@lex.COMMA), ref(expr_ref)).map(
        fn {
          [x] => x
          tuple => S::Tuple(tuple)
        },
      ) <<
      tok(@lex.RPAREN)
    ) +
    block_expr +
    bool.map(S::Bool) +
    (tok(@lex.SUB) >> ref(value_ref).map(fn { x => S::Neg(x, kind=None) })) +
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
    ) +
    id.map(S::Var)

  // Get or apply
  let get_or_apply_postfix : Parser[GetApplyPostFix] = (
      tok(@lex.LBKT) >>
      ref(expr_ref).map(GetApplyPostFix::Index) <<
      tok(@lex.RBKT)
    ) +
    (
      tok(@lex.LPAREN) >>
      sep_list(tok(@lex.COMMA), ref(expr_ref)).map(GetApplyPostFix::Apply) <<
      tok(@lex.RPAREN)
    )
  let get_or_apply : PS = lift2(
    merge_postfix,
    ref(value_ref),
    many(get_or_apply_postfix),
  )

  // If
  let if_ : PS = get_or_apply +
    lift3(
      fn(_test, _then, _else) { S::If(_test, _then, _else) },
      tok(@lex.IF) >> ref(expr_ref),
      block_expr,
      (tok(@lex.ELSE) >> block_expr) + epsilon().map(fn { _ => S::Unit }),
    )

  // Mul
  let mul_op = tok(@lex.MUL).map(fn { _ => B::Mul }) +
    tok(@lex.DIV).map(fn { _ => B::Div })
  let mul_ : PS = tier_left(mul_op, if_)

  // Add
  let add_op = tok(@lex.ADD).map(fn { _ => B::Add }) +
    tok(@lex.SUB).map(fn { _ => B::Sub })
  let add_ : PS = tier_left(add_op, mul_)

  // Cmp
  let cmp_op = tok(@lex.LE).map(
      fn { _ => fn(lhs : S, rhs : S) { S::LE(lhs, rhs) } },
    ) +
    tok(@lex.EQ).map(fn { _ => fn(lhs : S, rhs : S) { S::Eq(lhs, rhs) } })
  let cmp_ : PS = lift3(fn(lhs, op, rhs) { op(lhs, rhs) }, add_, cmp_op, add_) +
    add_
  expr_ref.val = cmp_

  // Statement
  let type_annotation = maybe(tok(@lex.COLON) >> ref(type_ref))
  let param = lift2(
    fn(id : String, ty : T?) { (id, fix_type(ty)) },
    id,
    type_annotation,
  )
  let nontop_head = lift2_maybe(
      fn(get_expr, rhs) {
        match get_expr {
          S::Get(_, _) => ()
          _ => return None
        }
        Some(fn(rest) { S::Put(get_expr, rhs, rest) })
      },
      get_or_apply,
      tok(@lex.ASSIGN) >> ref(expr_ref),
    ) +
    lift4(
      fn(id : String, args : Array[(String, T)], ret_ty : T?, body : S) {
        fn(rest) {
          S::LetRec({ name: (id, fix_type(ret_ty)), args, body }, rest)
        }
      },
      tok(@lex.FN) >> id,
      tok(@lex.LPAREN) >> sep_list(tok(@lex.COMMA), param) << tok(@lex.RPAREN),
      maybe(tok(@lex.ARROW) >> ref(type_ref)),
      block_expr,
    ) +
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
          None => ids.map(fn { name => (name, T::Var({ val: None })) })
        }
        Some(fn(rest) { S::LetTuple(ids_typed, rhs, rest) })
      },
      tok(@lex.LET) >>
      tok(@lex.LPAREN) >>
      sep_list1(tok(@lex.COMMA), id) <<
      tok(@lex.RPAREN),
      type_annotation,
      tok(@lex.ASSIGN) >> ref(expr_ref),
    )
  let common_head = lift3(
    fn(id, ty, body) { fn(rest) { S::Let((id, fix_type(ty)), body, rest) } },
    tok(@lex.LET) >> id,
    type_annotation,
    tok(@lex.ASSIGN) >> ref(expr_ref),
  )
  stmt_ref.val = fold_right(
    ref(expr_ref),
    common_head + nontop_head,
    tok(@lex.SEMICOL).map(fn { _ => fn(u, v) { u(v) } }),
  )
  let top_head = lift2(
      fn(id : String, body : S) {
        fn(rest) { S::LetRec({ name: (id, T::Unit), args: [], body }, rest) }
      },
      tok(@lex.FN) >> id_p(fn(s) { s == "main" || s == "init" }),
      block_expr,
    ) +
    lift4(
      fn(id : String, args : Array[(String, T)], ret_ty : T?, body : S) {
        fn(rest) {
          S::LetRec({ name: (id, fix_type(ret_ty)), args, body }, rest)
        }
      },
      tok(@lex.FN) >> id_p(fn(s) { s != "main" && s != "init" }),
      tok(@lex.LPAREN) >> sep_list(tok(@lex.COMMA), param) << tok(@lex.RPAREN),
      maybe(tok(@lex.ARROW) >> ref(type_ref)),
      block_expr,
    )
  fold_right(
    tok(@lex.EOF).map(fn { _ => S::Unit }),
    // NOTE: top level definitions must start at the beginning of line
    line_start >> (top_head + common_head),
    tok(@lex.SEMICOL).map(fn { _ => fn(u, v) { u(v) } }),
  )
}
