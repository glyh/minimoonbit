fn id_to_ast!(cst : ID) -> String {

}

fn ty_to_ast!(cst : Type) -> @types.Type {

}

fn exp_to_ast!(cst : Exp) -> @types.Syntax {

}

fn stmt_to_ast!(cst : Stmt) -> @types.Syntax {

}

fn args_to_ast!(
  cst : Result[Array[(ID, Type)], ParseError]
) -> Array[(String, @types.Type)] {

}

fn cst_to_ast!(cst : Program) -> @types.Syntax {
  let mut out : @types.Syntax = Unit
  for topi in cst._.rev().iter() {
    match topi._ {
      Err(e) => raise e
      Ok((LetDecl(id, ty, exp), _)) => {
        let id = id_to_ast!(id)
        let _ty = ty_to_ast!(ty)
        let rhs = exp_to_ast!(exp)
        out = Let((id, _ty), rhs, out)
      }
      Ok((FnDecl(id, args, ret_ty, body), _)) => {
        let name = id_to_ast!(id)
        let args = args_to_ast!(args)
        let ret_ty = ty_to_ast!(ret_ty)
        let body = stmt_to_ast!(body)
        out = LetRec({ name: (name, ret_ty), args, body }, out)
      }
      Ok((MainDecl(body), _)) => {
        let name = "main"
        let args = []
        let ret_ty = @types.Type::Unit
        let body = stmt_to_ast!(body)
        out = LetRec({ name: (name, ret_ty), args, body }, out)
      }
      Ok((InitDecl(body), _)) => {
        let name = "init"
        let args = []
        let ret_ty = @types.Type::Unit
        let body = stmt_to_ast!(body)
        out = LetRec({ name: (name, ret_ty), args, body }, out)
      }
    }
  }
  out
}

pub fn parse_string(s : string) -> @types.Syntax? {
  let tokens = try {
    @lex.lex_string!(s)
  } catch {
    LexError(e) => {
      println("!Lex: \{e}.")
      return None
    }
  }
  let program_parsed = program(tokens[:])
  for e in recoverable_errors.iter() {
    println("!Parse: \{e}.")
  }

  //try {
  //
  //
  //}
  //program()
}
