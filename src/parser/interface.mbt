fn is_concrete(t : T) -> Bool {
  match t {
    Var(_) => false
    _ => true
  }
}

fn ensure_top_level_type_sig(program : S) -> Unit {
  loop program {
    Let((_, ty), _, inner) => {
      if not(is_concrete(ty)) {
        @util.die("Need outer level type annotation for \{program}")
      }
      continue inner
    }
    LetRec({ name, args, .. }, inner) => {
      if not(args.map(fn { (_, ty) => ty }).iter().all(is_concrete)) ||
        not(is_concrete(name.1)) {
        @util.die("Need outer level type annotation for \{program}")
      }
      continue inner
    }
    _ => break
  }
}

pub fn parse_program(intput : String) -> S {
  let lex_stream = try {
    @lex.lex_string!(intput)
  } catch {
    LexError(e) => @util.die("Lexer Error: \{e}")
  }
  let parsed = match parser().parse(lex_stream[:]) {
    (Some(parsed), []) => parsed
    _ => @util.die("Parser Error")
  }
  ensure_top_level_type_sig(parsed)
  parsed
}
