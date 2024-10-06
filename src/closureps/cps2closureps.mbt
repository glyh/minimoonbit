// replace any function vars 
fn CloEnv::rebind_var(self : CloEnv, v : Var) -> Var {
  match self.bindings[v] {
    None => v
    Some(wrapped) => wrapped
  }
}

fn CloEnv::rebind_value(self : CloEnv, v : Value) -> Value {
  match v {
    Var(v) => Var(self.rebind_var(v))
    v => v
  }
}

// collect all closures to top level and fix call convention
fn CloEnv::collect_closure(self : CloEnv, s : S) -> S {
  fn rec(c : S) {
    self.collect_closure(c)
  }

  fn recrbva(v : Value) {
    self.rebind_value(v)
  }

  match s {
    Tuple(record, bind, rest) =>
      Tuple(
        record.map(recrbva),
        bind,
        // NOTE: the reason for add all bindings is to shadow any closure rebind
        // so it doesn't accidentally rebind too muach than it should
        self.add_rebind(bind, bind).collect_closure(rest),
      )
    KthTuple(idx, v, bind, rest) =>
      KthTuple(idx, v, bind, self.add_rebind(bind, bind).collect_closure(rest))
    Switch(v, branches) => Switch(v, branches.map(rec))
    Prim(op, args, bind, rest) =>
      Prim(op, args, bind, self.add_rebind(bind, bind).collect_closure(rest))
    Fix(f, args, body, rest) => {
      // Step 1. Calculate free variables of body
      let fvs = body.free_variables()
      fvs.remove(f)
      args.each(fn { a => fvs.remove(a) })
      let free_vars = fvs.iter().collect()

      // Step 2. Calculate the free variable tuple we need to pass inside the
      // closure 
      let fv_data_ty = match free_vars {
        [] => T::Unit
        _ => T::Tuple(free_vars.map(fn { v => v.ty }))
      }

      // this is the closure passed into the function
      let closure_ref = self.new_named(
        "closure_ref_\{f.to_string()}",
        T::Tuple([f.ty, fv_data_ty]),
      )

      // fix the type of f to accept an additional closure arg at the end
      guard let T::Fun(args_ty, _) = f.ty else {
        _ => @util.die("calling non function")
        // NOTE: the following alters f's type 
      }
      args_ty.push(closure_ref.ty) // WARN: after this operation our ds is now self-recursive
      let body_fixed = match free_vars {
        [] => rec(body)
        _ => {
          let fn_ptr = self.new_named("fn_ptr", f.ty)
          let freevars = self.new_named("freevars", fv_data_ty)
          let body_to_wrap = self
            .add_rebind(f, closure_ref)
            .collect_closure(body)
          let body_with_freevars_bound = free_vars.foldi(
            init=body_to_wrap,
            fn(idx, acc, ele) { KthTuple(idx, Var(freevars), ele, acc) },
          )
          KthTuple(
            0,
            Var(closure_ref),
            fn_ptr,
            KthTuple(1, Var(closure_ref), freevars, body_with_freevars_bound),
          )
        }
      }
      self.fnblocks[f] = { args, free_vars, body: body_fixed }
      let freevars_captured = self.new_named("freevars_captured", fv_data_ty)
      let closure_gen = self.new_named(
        "closure_\{f.to_string()}",
        T::Tuple([f.ty, fv_data_ty]),
      )
      let rest_fixed = self.add_rebind(f, closure_gen).collect_closure(rest)
      Tuple(
        free_vars.map(Value::Var),
        freevars_captured,
        Tuple([Label(f), Var(freevars_captured)], closure_gen, rest_fixed),
      )
    }
    App(f, args) =>
      match f {
        Var(f_var) =>
          // NOTE: always generate a call as if we're calling a closure. 
          // Since there's no way for us to decide whether we're calling a 
          // closure or not.
          match self.bindings[f_var] {
            Some(maybe_closure) => {
              args.push(Var(maybe_closure))
              // we know the called function statically, so we're allowed to mark 
              // it as a label
              App(Label(f_var), args)
            }
            None => {
              args.push(f)
              App(f, args)
            }
          }
        // NOTE: must be a native call
        // there's no guarantee we always use this case for all native calls
        Label(_) => App(f, args.map(recrbva))
        _ => @util.die("Can't invoke call on \{f}")
      }
    Just(v) => Just(recrbva(v))
  }
}
