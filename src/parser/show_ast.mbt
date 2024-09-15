impl Show for ID with output(id, logger) {
  logger.write_string(
    match id._ {
      Ok((name, _)) => name
      Err(_) => "!"
    },
  )
}

impl Show for Value with output(v, logger) {
  match v {
    I32(i) => Show::output(i, logger)
    F64(f) => Show::output(f, logger)
    Bool(b) => Show::output(b, logger)
    Unit => logger.write_string("()")
  }
}

impl Show for Binop with output(bop, logger) {
  match bop._ {
    Err(_) => logger.write_string("<!>")
    Ok((Eq, _)) => logger.write_string("==")
    Ok((Le, _)) => logger.write_string("<=")
    Ok((Add, _)) => logger.write_char('+')
    Ok((Sub, _)) => logger.write_char('-')
    Ok((Mul, _)) => logger.write_char('*')
    Ok((Div, _)) => logger.write_char('/')
  }
}

impl Show for Exp with output(__exp, logger) {
  match __exp._ {
    Err(_) => logger.write_char('!')
    Ok((Atom(v), _)) => Show::output(v, logger)
    Ok((Tuple(exps), _)) => {
      logger.write_char('(')
      loop exps[:] {
        [exp] => {
          Show::output(exp, logger)
          logger.write_char(')')
          break
        }
        [exp, .. as rest] => {
          Show::output(exp, logger)
          logger.write_char(',')
          continue rest
        }
        [] => break logger.write_char(')')
      }
    }
    Ok((Unary(Not, inner), _)) => {
      logger.write_string("not(")
      Show::output(inner, logger)
      logger.write_string(")")
    }
    Ok((Unary(Neg, inner), _)) => {
      logger.write_string("-")
      Show::output(inner, logger)
    }
    Ok((Binary(bop, lhs, rhs), _)) => {
      logger.write_char('(')
      Show::output(lhs, logger)
      Show::output(bop, logger)
      Show::output(rhs, logger)
      logger.write_char(')')
    }
    Ok((Apply(f, args), _)) => {
      Show::output(f, logger)
      logger.write_char('(')
      loop args[:] {
        [arg] => {
          Show::output(arg, logger)
          logger.write_char(')')
          break
        }
        [arg, .. as rest] => {
          Show::output(arg, logger)
          logger.write_char(',')
          continue rest
        }
        [] => break logger.write_char(')')
      }
    }
    Ok((If(cond, then_branch, else_branch), _)) => {
      logger.write_string("if ")
      Show::output(cond, logger)
      logger.write_string(" {")
      Show::output(then_branch, logger)
      logger.write_string("} else {")
      Show::output(else_branch, logger)
      logger.write_string("}")
    }
    Ok((Block(stmt), _)) => {
      logger.write_string("{")
      Show::output(stmt, logger)
      logger.write_string("}")
    }
    Ok((Var(id), _)) => Show::output(id, logger)
    Ok((Index(exp, accessor), _)) => {
      Show::output(exp, logger)
      logger.write_string("[")
      Show::output(accessor, logger)
      logger.write_string("]")
    }
    Ok((MakeArray(len, ele), _)) => {
      logger.write_string("Array::make(")
      Show::output(len, logger)
      logger.write_string(",")
      Show::output(ele, logger)
      logger.write_string(")")
    }
  }
}

fn maybe_show_type_ann(
  ty : Type?,
  logger : Logger,
  ~prefix : String = ": "
) -> Unit {
  match ty {
    Some(ty) => {
      logger.write_string(prefix)
      Show::output(ty, logger)
    }
    None => ()
  }
}

impl Show for Stmt with output(stmt, logger) {
  match stmt._ {
    Err(_) => logger.write_string("!")
    Ok((heads, last_exp, _)) => {
      for head in heads.iter() {
        match head._ {
          Err(_) => logger.write_string("!;")
          Ok((Assign(lhs, rhs), _)) => {
            Show::output(lhs, logger)
            logger.write_string(" = ")
            Show::output(rhs, logger)
          }
          Ok((NontopFn(name, params, return_ty, body), _)) => {
            logger.write_string("fn ")
            Show::output(name, logger)
            logger.write_string("(")
            loop params[:] {
              [] => break
              [(id, maybe_type)] => {
                Show::output(id, logger)
                maybe_show_type_ann(maybe_type, logger)
                break
              }
              [(id, maybe_type), .. as rest] => {
                Show::output(id, logger)
                maybe_show_type_ann(maybe_type, logger)
                logger.write_string(", ")
                continue rest
              }
            }
            logger.write_string(") ")
            maybe_show_type_ann(return_ty, logger, prefix="-> ")
            logger.write_string("{")
            Show::output(body, logger)
            logger.write_string("}; ")
          }
          Ok((Let(name, ty, rhs), _)) => {
            logger.write_string("let ")
            Show::output(name, logger)
            maybe_show_type_ann(ty, logger)
            logger.write_string("= ")
            Show::output(rhs, logger)
            logger.write_string("; ")
          }
          Ok((LetTuple(tup, ty, rhs), _)) => {
            logger.write_string("let (")
            loop tup[:] {
              [] => break
              [id] => break Show::output(id, logger)
              [id, .. as rest] => {
                Show::output(id, logger)
                logger.write_string(", ")
                continue rest
              }
            }
            logger.write_string(")")
            maybe_show_type_ann(ty, logger)
            logger.write_string("= ")
            Show::output(rhs, logger)
            logger.write_string("; ")
          }
        }
      }
      Show::output(last_exp, logger)
    }
  }
}

impl Show for Type with output(ty, logger) {
  match ty._ {
    Err(_) => logger.write_string("!T")
    Ok((TBool, _)) => logger.write_string("Bool")
    Ok((TUnit, _)) => logger.write_string("Unit")
    Ok((TInt, _)) => logger.write_string("Int")
    Ok((TDouble, _)) => logger.write_string("Double")
    Ok((TArray(inner), _)) => {
      logger.write_string("Array[")
      Show::output(inner, logger)
      logger.write_string("]")
    }
    Ok((TTuple(inners), _)) => {
      logger.write_string("(")
      loop inners[:] {
        [] => break
        [single] => break Show::output(single, logger)
        [hd, .. as rest] => {
          Show::output(hd, logger)
          logger.write_string(", ")
          continue rest
        }
      }
      logger.write_string(")")
    }
    Ok((TFunction(args, return_ty), _)) => {
      logger.write_string("(")
      loop args[:] {
        [] => break
        [single] => break Show::output(single, logger)
        [hd, .. as rest] => {
          Show::output(hd, logger)
          logger.write_string(", ")
          continue rest
        }
      }
      logger.write_string(") -> ")
      Show::output(return_ty, logger)
    }
  }
}

impl Show for Program with output(prog, logger) { Show::output(prog._, logger) }

impl Show for Top with output(top, logger) {
  match top._ {
    Err(_) => logger.write_string("!")
    Ok((LetDecl(id, ty, exp), _)) => {
      logger.write_string("let ")
      Show::output(id, logger)
      maybe_show_type_ann(Some(ty), logger)
      logger.write_string(" = ")
      Show::output(exp, logger)
    }
    Ok((MainDecl(exp), _)) => {
      logger.write_string("fn main {")
      Show::output(exp, logger)
      logger.write_string("}")
    }
    Ok((InitDecl(exp), _)) => {
      logger.write_string("fn init {")
      Show::output(exp, logger)
      logger.write_string("}")
    }
    Ok((FnDecl(name, args, return_ty, body), _)) => {
      logger.write_string("fn ")
      Show::output(name, logger)
      logger.write_string("(")
      match args {
        Err(_) => logger.write_string("!")
        Ok(args) =>
          loop args[:] {
            [] => break
            [arg] => break Show::output(arg, logger)
            [arg, .. as args] => {
              Show::output(arg, logger)
              continue args
            }
          }
      }
      logger.write_string(")")
      maybe_show_type_ann(Some(return_ty), logger, prefix=" -> ")
      logger.write_string(" {")
      Show::output(body, logger)
      logger.write_string("}")
    }
  }
}
