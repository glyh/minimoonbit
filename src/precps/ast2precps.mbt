pub fn TyEnv::ast2precps(self : TyEnv, s : S) -> PreCps {
  fn rec(s : S) {
    self.ast2precps(s)
  }

  match s {
    Unit => Unit
    Bool(true) => Int(1)
    Bool(false) => Int(0)
    Int(i) => Int(i)
    Double(f) => Double(f)
    Var(name) =>
      match self.find(name) {
        Some(v) => v
        None => @util.die("No binding for \{name}")
      }
    Tuple(tup) => {
      let tup_outs = tup.map(rec)
      let tys = tup_outs.map(fn { p => p.get_type() })
      let tup_ty = T::Tuple(tys)
      Tuple(tup_ty, tup_outs)
    }
    Not(inner) => Prim(Bool, Not, [rec(inner)])
    Array(len, elem) => {
      let elem = rec(elem)
      let len = rec(len)
      let ret_ty = T::Array(elem.get_type())
      Prim(ret_ty, MakeArray, [len, elem])
    }
    Neg(inner, ~kind) => {
      let n : Numeric = if kind == T::Int { Int } else { Double }
      Prim(kind, Neg(n), [rec(inner)])
    }
    App(f, args) => {
      let f = rec(f)
      guard let Fun(_, ret_ty) = f.get_type() else {
        _ => @util.die("Calling a non function")
      }
      App(ret_ty, f, args.map(rec))
    }
    Get(arr, idx) => {
      let arr = rec(arr)
      guard let Array(ele_ty) = arr.get_type() else {
        _ => @util.die("indexing a non array")
      }
      Prim(ele_ty, Get, [arr, rec(idx)])
    }
    If(cond, _then, _else) => {
      let cond = rec(cond)
      let _then = rec(_then)
      let _else = rec(_else)
      If(_then.get_type(), cond, _then, _else)
    }
    Prim(lhs, rhs, op, ~kind) => {
      let n : Numeric = if kind == T::Int { Int } else { Double }
      Prim(kind, Math(op, n), [rec(lhs), rec(rhs)])
    }
    Eq(lhs, rhs) => Prim(Bool, Eq, [rec(lhs), rec(rhs)])
    LE(lhs, rhs) => Prim(Bool, Le, [rec(lhs), rec(rhs)])
    Let((name, ty), rhs, rest) => {
      let (bind, env_new) = self.add(name, ty)
      Let(ty, bind, rec(rhs), env_new.ast2precps(rest))
    }
    LetRec(f, rest) => {
      // NOTE: we don't actually need any extra variables for simply refer to the function pointer once we've done closure conversion
      let (fvar, env_rest) = self.add_label(f.name.0, f.name.1)
      let mut env_body = env_rest
      let args = []
      f.args.each(
        fn(arg) {
          let (name, ty) = arg
          let (argvar, env_body_new) = env_body.add(name, ty)
          args.push(argvar)
          env_body = env_body_new
        },
      )
      LetRec(
        f.name.1,
        fvar,
        args,
        env_body.ast2precps(f.body),
        env_rest.ast2precps(rest),
      )
    }
    LetTuple(tup, rhs, rest) => {
      let tup_ty = T::Tuple(tup.map(fn { (_, ty) => ty }))
      let tup_var = self.gen_tmp()
      fn go(
        tup : ArrayView[(String, T)],
        idx : Int,
        env_cur : TyEnv
      ) -> PreCps {
        match tup {
          [] => env_cur.ast2precps(rest)
          [ti, .. as tup_rest] => {
            let (vari, env_next) = env_cur.add(ti.0, ti.1)
            Let(
              ti.1,
              vari,
              KthTuple(ti.1, idx, Var(tup_ty, tup_var)),
              go(tup_rest, idx + 1, env_next),
            )
          }
        }
      }

      Let(tup_ty, tup_var, rec(rhs), go(tup[:], 0, self))
    }
    Put(arr, idx, rhs) => Prim(Unit, Put, [rec(arr), rec(idx), rec(rhs)])
  }
}
