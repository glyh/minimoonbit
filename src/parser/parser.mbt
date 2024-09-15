// According to https://taolun.moonbitlang.com/t/topic/882/2, we don't have GADT, so I have to hand roll like below

let recoverable_errors : Array[ParseError] = []

fn program(view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) -> Program {
  let program : Program = []
  let mut view = view
  while true {
    match view {
      [(@lex.FN, all_start, _) as tok, ..] => {
        if all_start.col != 1 { // in grammar file it's enforced all top levels start at the beginning of line
          recoverable_errors.push(TopLevelNotColNot1(tok))
        }
        let (top_fn, rest) = top_fn(view)
        program._.push(top_fn)
        view = rest
      }
      [(@lex.LET, all_start, _) as tok, ..] => {
        if all_start.col != 1 {
          recoverable_errors.push(TopLevelNotColNot1(tok))
        }
        let (top_let, rest) = top_let(view)
        program._.push(top_let)
        view = rest
      }
      [tok, ..] => program._.push(Err(Unexpected(tok)))
      [] => break
    }
  }
  program
}

fn consume(
  tok_expect : @lex.Token,
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)] // returns the end if consume
) -> (@lex.Loc?, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  match view {
    [(tok, tok_s, tok_e), .. as view] =>
      if tok == tok_expect {
        (Some(tok_e), view)
      } else {
        recoverable_errors.push(Missing(tok, tok_s))
        (None, view)
      }
    [] => {
      recoverable_errors.push(ExpectButEOF(tok_expect))
      (None, view)
    }
  }
}

fn consume_head(
  tok_expect : @lex.Token,
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)] // returns the end if consume
) -> (@lex.Loc?, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  match view {
    [(tok, tok_s, _), .. as view] =>
      if tok == tok_expect {
        (Some(tok_s), view)
      } else {
        recoverable_errors.push(Missing(tok, tok_s))
        (None, view)
      }
    [] => {
      recoverable_errors.push(ExpectButEOF(tok_expect))
      (None, view)
    }
  }
}

fn top_fn(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Top, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  guard let (Some(all_start), view) = consume_head(@lex.FN, view) else {
    _ => return (Err(Unreachable), view)
  }
  let (name, view) = match view {
    [(@lex.ID(id), id_s, id_e), .. as view] =>
      ((id, { src_start: id_s, src_end: Some(id_e) }), view)
    [(@lex.MAIN, id_s, id_e), .. as view] =>
      (("main", { src_start: id_s, src_end: Some(id_e) }), view)
    [(@lex.INIT, id_s, id_e), .. as view] =>
      (("init", { src_start: id_s, src_end: Some(id_e) }), view)
    [tok, .. as view] => return (Err(Unexpected(tok)), view)
    [] => return (Err(EarlyEOF), view)
  }
  if name.0 == "main" {
    let (_, view) = consume(@lex.LBRACE, view)
    let (fn_body, view) = stmt(view) // { body }
    let (_, view) = consume(@lex.RBRACE, view)
    let (all_end, view) = consume(@lex.SEMICOL, view)
    (Ok((MainDecl(fn_body), { src_start: all_start, src_end: all_end })), view)
  } else if name.0 == "init" {
    let (_, view) = consume(@lex.LBRACE, view)
    let (fn_body, view) = stmt(view) // { body }
    let (_, view) = consume(@lex.RBRACE, view)
    let (all_end, view) = consume(@lex.SEMICOL, view)
    (Ok((InitDecl(fn_body), { src_start: all_start, src_end: all_end })), view)
  } else {
    let (params, view) = params_typed(view) // (x1, x2, x3)
    let (_, view) = consume(@lex.ARROW, view)
    let (return_type, view) = _type(view) // Type
    let (_, view) = consume(@lex.LBRACE, view)
    let (fn_body, view) = stmt(view)
    let (_, view) = consume(@lex.RBRACE, view)
    let (all_end, view) = consume(@lex.SEMICOL, view)
    (
      Ok(
        (
          FnDecl(Ok(name), params, return_type, fn_body),
          { src_start: all_start, src_end: all_end },
        ),
      ),
      view,
    )
  }
}

fn id(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (ID, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  match view {
    [(@lex.ID(id), id_s, id_e), .. as view] =>
      (Ok((id, { src_start: id_s, src_end: Some(id_e) })), view)
    [tok, ..] => (Err(Unexpected(tok)), view)
    [] => (Err(EarlyEOF), view)
  }
}

fn top_let(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Top, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  guard let (Some(all_start), view) = consume_head(@lex.LET, view) else {
    _ => return (Err(Unreachable), view)
  }
  let (name, view) = id(view)
  let (_, view) = consume(@lex.COLON, view)
  let (_type, view) = _type(view)
  let (_, view) = consume(@lex.EQ, view)
  let (expr, view) = expression(view)
  let (all_end, view) = consume(@lex.SEMICOL, view)
  (
    Ok((LetDecl(name, _type, expr), { src_start: all_start, src_end: all_end })),
    view,
  )
}

fn params_typed(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) ->
     (
       Result[Array[(ID, Type)], ParseError],
       ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)],
     ) {
  let view = match view {
    [(@lex.LPAREN, _, _), .. as view] => view
    [tok, .. as view] => return (Err(Unexpected(tok)), view)
    [] => return (Err(EarlyEOF), view)
  }
  let result : Array[(ID, Type)] = []
  loop view {
    [(@lex.RPAREN, _, _), .. as view] => break (Ok(result), view)
    [(@lex.ID(_), _, _), ..] => {
      let (name, view) = id(view)
      let (_, view) = consume(@lex.COLON, view)
      let (_type, view) = _type(view)
      result.push((name, _type))
      continue view
    }
    [tok, .. as view] => break (Err(Unexpected(tok)), view)
    [] => break (Err(ExpectButEOF(@lex.RPAREN)), view)
  }
}

fn take_end_maybe[T](v : Result[(T, Ann), ParseError]) -> @lex.Loc? {
  match v {
    Ok((_, { src_end: Some(maybe_end), .. })) => Some(maybe_end)
    _ => None
  }
}

//type Type Result[(TypeNaked, Ann), ParseError]
fn _type(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Type, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  match view {
    [(@lex.UNIT_T, start, end), .. as rest] =>
      (Ok((TUnit, { src_start: start, src_end: Some(end) })), rest)
    [(@lex.BOOL_T, start, end), .. as rest] =>
      (Ok((TBool, { src_start: start, src_end: Some(end) })), rest)
    [(@lex.INT_T, start, end), .. as rest] =>
      (Ok((TInt, { src_start: start, src_end: Some(end) })), rest)
    [(@lex.DOUBLE_T, start, end), .. as rest] =>
      (Ok((TDouble, { src_start: start, src_end: Some(end) })), rest)
    [(@lex.ARRAY, start, _), .. as view] => {
      let (_, view) = consume(@lex.LBKT, view)
      let (type_inner, view) = _type(view)
      let type_inner = type_inner
      let (all_end, view) = consume(@lex.RBKT, view)
      let type_array = (
        TArray(type_inner),
        { src_start: start, src_end: all_end },
      )
      (Ok(type_array), view)
    }
    [(@lex.LPAREN, start, _), .. as view] => {
      let (type_i, view) = _type(view)
      let acc = [type_i]
      let (all_end, view) = loop view {
        [(@lex.RPAREN, _, all_end), .. as view] => break (Some(all_end), view)
        [(@lex.COMMA, _, _), .. as view] => {
          let (type_i, view) = _type(view)
          acc.push(type_i)
          continue view
        }
        [(tok, s, e), .. as view] => {
          acc.push(Err(Unexpected((tok, s, e))))
          break (None, view)
        }
        [] => {
          acc.push(Err(EarlyEOF))
          break (None, view)
        }
      }
      match view {
        [(@lex.ARROW, _, _), .. as view] => {
          let (return_ty, view) = _type(view)
          let all_end = take_end_maybe(return_ty._)
          (
            Ok(
              (
                TFunction(acc, return_ty),
                { src_start: start, src_end: all_end },
              ),
            ),
            view,
          )
        }
        view =>
          (Ok((TTuple(acc), { src_start: start, src_end: all_end })), view)
      }
    }
    [(tok, s, e), .. as view] => return (Err(Unexpected((tok, s, e))), view)
    [] => return (Err(EarlyEOF), view)
  }
}

fn stmt(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Stmt, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  let non_exp_stmts : Array[StmtHead] = []
  let src_start = match view {
    [(_, _all_start, _), ..] => _all_start
    [] => return (Err(EarlyEOF), view)
  }
  let (last_exp, rest_view) = loop view {
    [(@lex.LET, all_start, _), .. as view] =>
      match view {
        [(@lex.ID(_), _, _), ..] => {
          let (id, view) = id(view)
          let (ty, view) = match view {
            [(@lex.EQ, _, _), .. as view] => (None, view)
            view => {
              let (_, view) = consume(@lex.COLON, view)
              let (ty, view) = _type(view)
              let (_, view) = consume(@lex.EQ, view)
              (Some(ty), view)
            }
          }
          let (value_exp, view) = expression(view)
          let (all_end, view) = consume(@lex.SEMICOL, view)
          non_exp_stmts.push(
            Ok(
              (
                Let(id, ty, value_exp),
                { src_start: all_start, src_end: all_end },
              ),
            ),
          )
          continue view
        }
        [(@lex.LPAREN, _, _), .. as view] => {
          let id_tuples : Array[ID] = []
          let mut view = view
          while true {
            let (name, _view) = id(view)
            id_tuples.push(name)
            match _view {
              [(@lex.RPAREN, _, _), .. as _view] => {
                view = _view
                break
              }
              [(@lex.COMMA, _, _), .. as _view] => view = _view
              [(tok, s, e), .. as _view] => {
                id_tuples.push(Err(Unexpected((tok, s, e))))
                view = _view
                break
              }
              [] => {
                id_tuples.push(Err(EarlyEOF))
                break
              }
            }
          }
          let (ty, view) = match view {
            [(@lex.EQ, _, _), .. as view] => (None, view)
            view => {
              let (_, view) = consume(@lex.COLON, view)
              let (ty, view) = _type(view)
              let (_, view) = consume(@lex.EQ, view)
              (Some(ty), view)
            }
          }
          let (value_exp, view) = expression(view)
          let (all_end, view) = consume(@lex.SEMICOL, view)
          non_exp_stmts.push(
            Ok(
              (
                LetTuple(id_tuples, ty, value_exp),
                { src_start: all_start, src_end: all_end },
              ),
            ),
          )
          continue view
        }
        [tok, .. as view] => {
          non_exp_stmts.push(Err(Unexpected(tok)))
          continue view
        }
        [] => {
          non_exp_stmts.push(Err(EarlyEOF))
          break (Err(EarlyEOF), view)
        }
      }
    [(@lex.FN, all_start, _), .. as view] => {
      let (fn_id, view) = id(view)
      let (_, view) = consume(@lex.LPAREN, view)
      let params : Array[(ID, Type?)] = []
      let view = loop view {
        [(@lex.RPAREN, _, _), .. as view] => break view
        view => {
          let (param_id, view) = id(view)
          let (param_ty, view) = match view {
            [(@lex.RPAREN, _, _), ..] => (None, view) // don't consume RPAREN
            [(@lex.COMMA, _, _), .. as view] => (None, view)
            [] as view => {
              params.push((Err(EarlyEOF), None))
              break view
            }
            view => {
              let (_, view) = consume(@lex.COLON, view)
              let (ty, view) = _type(view)
              (Some(ty), view)
            }
          }
          params.push((param_id, param_ty))
          continue view
        }
      }
      let (return_ty, view) = match view {
        [(@lex.ARROW, _, _), .. as view] => {
          let (ty, view) = _type(view)
          (Some(ty), view)
        }
        view => (None, view)
      }
      let (_, view) = consume(@lex.LBRACE, view)
      let (stmt, view) = stmt(view)
      let (_, view) = consume(@lex.RBRACE, view)
      let (all_end, view) = consume(@lex.SEMICOL, view)
      //return (Err(EarlyEOF), view)
      non_exp_stmts.push(
        Ok(
          (
            NontopFn(fn_id, params, return_ty, stmt),
            { src_start: all_start, src_end: all_end },
          ),
        ),
      )
      continue view
    }
    [(_, all_start, _), ..] as view => {
      // consumes expression
      let (value_exp, view) = expression(view)
      match view {
        [(@lex.EQ, _, _), .. as view] => {
          let (rhs, view) = expression(view)
          let (all_end, view) = consume(@lex.SEMICOL, view)
          non_exp_stmts.push(
            Ok(
              (
                Assign(value_exp, rhs),
                { src_start: all_start, src_end: all_end },
              ),
            ),
          )
          continue view
        }
        view => break (value_exp._, view)
      }
    }
    [] => (Err(EarlyEOF), view)
  }
  let src_end = take_end_maybe(last_exp)
  (Ok((non_exp_stmts, last_exp, { src_start, src_end })), rest_view)
}

fn expression(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Exp, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  let sub = add_expr
  let all_start = match view {
    [(_, all_start, _), ..] => all_start
    _ => return (Err(EarlyEOF), view)
  }
  let (lhs, view) = sub(view)
  let (op_rel, view) : (Binop, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) = match
    view {
    [(@lex.EQ, op_s, op_e), .. as view] =>
      (Ok((Eq, { src_start: op_s, src_end: Some(op_e) })), view)
    [(@lex.LE, op_s, op_e), .. as view] =>
      (Ok((Le, { src_start: op_s, src_end: Some(op_e) })), view)
    _ => return (lhs, view)
  }
  let (rhs, view) = sub(view)
  let all_end = take_end_maybe(rhs._)
  (
    Ok((Binary(op_rel, lhs, rhs), { src_start: all_start, src_end: all_end })),
    view,
  )
}

fn add_expr(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Exp, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  let sub = mul_expr
  let all_start = match view {
    [(_, all_start, _), ..] => all_start
    _ => return (Err(EarlyEOF), view)
  }
  let (lhs, view) = sub(view)
  let mut lhs = lhs
  let mut view = view
  while true {
    let (op_rel, _view) : (Binop, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) = match
      view {
      [(@lex.ADD, op_s, op_e), .. as _view] =>
        (Ok((Add, { src_start: op_s, src_end: Some(op_e) })), _view)
      [(@lex.SUB, op_s, op_e), .. as _view] =>
        (Ok((Sub, { src_start: op_s, src_end: Some(op_e) })), _view)
      _ => return (lhs, view)
    }
    let (rhs, _view) = sub(_view)
    view = _view
    let all_end = take_end_maybe(rhs._)
    lhs = Ok(
      (Binary(op_rel, lhs, rhs), { src_start: all_start, src_end: all_end }),
    )
  }
  (lhs, view) // this is not used at all but the type system requires it to function well
}

fn mul_expr(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Exp, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  let sub = if_expr
  let all_start = match view {
    [(_, all_start, _), ..] => all_start
    _ => return (Err(EarlyEOF), view)
  }
  let (lhs, view) = sub(view)
  let mut lhs = lhs
  let mut view = view
  while true {
    let (op_rel, _view) : (Binop, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) = match
      view {
      [(@lex.MUL, op_s, op_e), .. as _view] =>
        (Ok((Mul, { src_start: op_s, src_end: Some(op_e) })), _view)
      [(@lex.DIV, op_s, op_e), .. as _view] =>
        (Ok((Div, { src_start: op_s, src_end: Some(op_e) })), _view)
      _ => return (lhs, view)
    }
    let (rhs, _view) = sub(_view)
    view = _view
    let all_end = take_end_maybe(rhs._)
    lhs = Ok(
      (Binary(op_rel, lhs, rhs), { src_start: all_start, src_end: all_end }),
    )
  }
  (lhs, view) // this is not used at all but the type system requires it to function well
}

test "how you return from a while" {
  fn t() -> Int {
    while true {
      return 1
    }
    2
  }

  assert_eq!(t(), 1)
}

fn block_expr(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Stmt, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  let (_, view) = consume(@lex.LBRACE, view)
  let (_stmt, view) = stmt(view)
  let (_, view) = consume(@lex.RBRACE, view)
  (_stmt, view)
}

fn if_expr(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Exp, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  match view {
    [(@lex.IF, all_start, _), .. as view] => {
      let (cond, view) = expression(view)
      let (then_clause, view) = block_expr(view)
      let (else_clause, view) = match view {
        [(@lex.ELSE, _, _), .. as view] => {
          let (else_clause, view) = block_expr(view)
          (Some(else_clause), view)
        }
        view => (None, view)
      }
      let naked = If(cond, then_clause, else_clause)
      (Ok((naked, { src_start: all_start, src_end: None })), view)
    }
    view => get_apply_expr(view)
  }
}

fn get_apply_expr(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Exp, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  let all_start = match view {
    [(_, all_start, _), ..] => all_start
    _ => return (Err(EarlyEOF), view)
  }
  let (_value, view) = value_expr(view)
  let mut acc = _value
  let mut view = view
  while true {
    match view {
      [(@lex.LBKT, _, _), .. as _view] => {
        let (accessor, _view) = expression(_view)
        let (cur_end, _view) = consume(@lex.RBKT, _view)
        acc = Ok(
          (Index(acc, accessor), { src_start: all_start, src_end: cur_end }),
        )
        view = _view
      }
      [(@lex.LPAREN, _, _), .. as _view] => {
        let params : Array[Exp] = []
        let mut cur_end = None
        match _view {
          [(@lex.RPAREN, _, _end), .. as _view] => {
            cur_end = Some(_end)
            view = _view
          }
          _view =>
            while true {
              let (exp, _view) = expression(_view)
              params.push(exp)
              match _view {
                [(@lex.COMMA, _, _), .. as _view] => {
                  view = _view
                  continue
                }
                _view => {
                  // otherwise we assume we hit the end
                  let (_end, _view) = consume(@lex.RPAREN, _view)
                  cur_end = _end
                  view = _view
                  break
                }
              }
            }
        }
        acc = Ok(
          (Apply(acc, params), { src_start: all_start, src_end: cur_end }),
        )
      }
      _ => break
    }
  }
  (acc, view)
}

fn value_expr(
  view : ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]
) -> (Exp, ArrayView[(@lex.Token, @lex.Loc, @lex.Loc)]) {
  match view {
    [(@lex.LPAREN, all_start, _), (@lex.RPAREN, _, all_end), .. as view] =>
      (Ok((Atom(Unit), { src_start: all_start, src_end: Some(all_end) })), view)
    [(@lex.LPAREN, all_start, _), .. as view] => {
      let (t_first, view) = expression(view)
      let tuple : Array[Exp] = [t_first]
      let mut view = view
      let mut all_end = None
      while true {
        match view {
          [(@lex.RPAREN, _, _all_end), .. as _view] => {
            all_end = Some(_all_end)
            view = _view
            break
          }
          _view => {
            let (_, _view) = consume(@lex.COMMA, _view)
            let (t_i, _view) = expression(view)
            tuple.push(t_i)
            view = _view
          }
        }
      }
      match tuple {
        [single] => (single, view)
        _ =>
          (Ok((Tuple(tuple), { src_start: all_start, src_end: all_end })), view)
      }
    }
    [(@lex.BOOL(b), all_start, all_end), .. as view] =>
      (
        Ok((Atom(Bool(b)), { src_start: all_start, src_end: Some(all_end) })),
        view,
      )
    [(@lex.ID(_), _, _), ..] as view => {
      let (_id, view) = id(view)
      //let dummy_ann = { src_start: @lex.new_loc(), src_end: None }
      //(Ok((Var(Ok(("wow", dummy_ann))), dummy_ann)), view)
      match _id._ {
        Ok((_, ann)) => (Ok((Var(_id), ann)), view)
        Err(err) => (Err(err), view)
      }
    }
    [(@lex.LBRACE, all_start, _), ..] as view => {
      let (stmt, rest) = block_expr(view)
      let all_end = match stmt._ {
        Ok(a) =>
          match a.1._ {
            Ok(b) => b.1.src_end
            _ => None
          }
        _ => None
      }
      match stmt._ {
        Err(e) => (Err(e), rest)
        Ok(_stmt) =>
          (Ok((Block(stmt), { src_start: all_start, src_end: all_end })), rest)
      }
    }
    [(@lex.SUB, all_start, _), .. as view] => {
      let (inner, view) = value_expr(view)
      let all_end = take_end_maybe(inner._)
      (
        Ok((Unary(Neg, inner), { src_start: all_start, src_end: all_end })),
        view,
      )
    }
    [(@lex.F64(f), all_start, all_end), .. as view] =>
      (
        Ok((Atom(F64(f)), { src_start: all_start, src_end: Some(all_end) })),
        view,
      )
    [(@lex.I32(i), all_start, all_end), .. as view] =>
      (
        Ok((Atom(I32(i)), { src_start: all_start, src_end: Some(all_end) })),
        view,
      )
    [(@lex.NOT, all_start, _), .. as view] => {
      let (_, view) = consume(@lex.LPAREN, view)
      let (inner, view) = expression(view)
      let (all_end, view) = consume(@lex.RPAREN, view)
      (
        Ok((Unary(Not, inner), { src_start: all_start, src_end: all_end })),
        view,
      )
    }
    //'Array' ':' ':' 'make' '(' expr ',' expr ')'; // Array::make(x, y)
    [(@lex.ARRAY, all_start, _), .. as view] => {
      let (_, view) = consume(@lex.COLON, view)
      let (_, view) = consume(@lex.COLON, view)
      let (_, view) = consume(@lex.MAKE, view)
      let (_, view) = consume(@lex.LPAREN, view)
      let (len, view) = expression(view)
      let (_, view) = consume(@lex.COMMA, view)
      let (ele, view) = expression(view)
      let (all_end, view) = consume(@lex.RPAREN, view)
      (
        Ok((MakeArray(len, ele), { src_start: all_start, src_end: all_end })),
        view,
      )
    }
    [(tok, s, e), .. as view] => (Err(Unexpected((tok, s, e))), view)
    [] => (Err(EarlyEOF), view)
  }
}
