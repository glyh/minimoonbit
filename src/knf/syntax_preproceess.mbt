enum DefoldPre {
  Let(String, Type, S)
  LetRec(F)
}

/// Preprocesses the syntax tree, extracting the contents of functions `main` and `init` to top-level.
pub fn KnfEnv::syntax_preprocess(_self : KnfEnv, s : S) -> S {
  let defolded : Array[DefoldPre] = []
  let mut main_body : S? = None
  let mut init_body : S? = None
  let used_names : @hashset.T[String] = @hashset.new()
  loop s {
    Unit => break
    LetRec({ name: (name, _), body, .. } as f, rest) => {
      if used_names.contains(name) {
        @util.die("Top level function \{name} defined twice!")
      } else {
        used_names.insert(name)
      }
      if name == "main" {
        main_body = Some(body)
      } else if name == "init" {
        init_body = Some(body)
      } else {
        defolded.push(LetRec(f))
      }
      continue rest
    }
    Let((name, ty), rhs, rest) => {
      if used_names.contains(name) {
        @util.die("Top level binding \{name} defined twice!")
      } else {
        used_names.insert(name)
      }
      defolded.push(Let(name, ty, rhs))
      continue rest
    }
    cur => @util.die("Unexpected top level \{cur}")
  }
  let inner = match (init_body, main_body) {
    (None, None) =>
      @util.die("Expect at least one of `init` and `main` in top level!")
    (Some(i), None) => i
    (None, Some(m)) => m
    (Some(i), Some(m)) => S::Let(("_", T::Unit), i, m)
  }
  fn merge_defolded(acc : S, ele : DefoldPre) -> S {
    match ele {
      Let(id, ty, rhs) => Let((id, ty), rhs, acc)
      LetRec(f) => LetRec(f, acc)
    }
  }

  defolded.rev_fold(init=inner, merge_defolded)
}
