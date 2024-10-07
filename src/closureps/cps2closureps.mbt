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
    Label(l) => Label(self.rebind_var(l))
    v => v
  }
}

// collect all closures to top level and fix call convention
// NOTE:
// whenever we store a reference to an external call, we need to wrap it as a closure
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
      // Step 1. recurse on body to collect free vars
      // HACK: this is the closure passed into the function
      // we set it's type to unit as we can't know it's type before collecting 
      // free vars, but to collect free var we have to have it bound
      let closure_ref = self.new_named("ref_\{f.to_string()}", T::Unit)
      let body_env = args
        .fold(init=self, fn(acc, ele) { acc.add_rebind(ele, ele) })
        .add_rebind(f, closure_ref)
      let mut body = body_env.collect_closure(body)

      // Step 2. Calculate free variables
      let free_vars = body
        .free_variables()
        .iter()
        .filter(fn(v) { v != closure_ref && not(args.contains(v)) })
        .collect()
      let has_free_vars = not(free_vars.is_empty())

      // Step 3. Rewrap the body on demand
      if has_free_vars {
        let fn_ptr = self.new_named("fn_ptr", f.ty.val)
        let fv_data_ty = T::Tuple(free_vars.map(fn { v => v.ty.val }))
        let freevars = self.new_named("fvs", fv_data_ty)
        body = free_vars.foldi(
          init=body,
          fn(idx, acc, ele) { KthTuple(idx, Var(freevars), ele, acc) },
        )
        body = KthTuple(
          0,
          Var(closure_ref),
          fn_ptr,
          KthTuple(1, Var(closure_ref), freevars, body),
        )
      }

      // Step 4. Patching f's args and types
      let fn_ptr_ty = f.ty.val
      let fv_data_ty = if has_free_vars {
        T::Tuple(free_vars.map(fn { v => v.ty.val }))
      } else {
        Unit
      }
      closure_ref.ty.val = T::Tuple([fn_ptr_ty, fv_data_ty])
      guard let T::Fun(_args_ty, ret_ty) = f.ty.val else {
        _ => @util.die("calling non function")
        // NOTE: the following alters f's type 
      }
      let args_ty = _args_ty.copy()
      args_ty.push(closure_ref.ty.val) // WARN: types are self-recursive now
      f.ty.val = Fun(args_ty, ret_ty)
      let args = args.copy()
      args.push(closure_ref)
      self.fnblocks[f] = { args, free_vars, body }

      // Step 4. recurse on rest

      let closure_gen = self.new_named(
        "clo_\{f.to_string()}",
        closure_ref.ty.val,
      )
      if has_free_vars {
        let freevars_captured = self.new_named("fvs_cap", fv_data_ty)
        let rest_fixed = self.add_rebind(f, closure_gen).collect_closure(rest)
        Tuple(
          free_vars.map(Value::Var),
          freevars_captured,
          Tuple([Label(f), Var(freevars_captured)], closure_gen, rest_fixed),
        )
      } else {
        let rest_fixed = self.add_rebind(f, closure_gen).collect_closure(rest)
        Tuple([Label(f), Unit], closure_gen, rest_fixed)
      }
    }
    App(f, args) =>
      match f {
        Var(f_var) =>
          match self.bindings[f_var] {
            Some(closure) => {
              args.push(Var(closure))
              // Calling a user level function
              let tmp = self.new_named("fn_ptr", f_var.ty.val)
              KthTuple(0, Var(closure), tmp, App(Var(tmp), args.map(recrbva)))
            }
            None => @util.die("undefined function \{f} called")
          }
        Label(f_var) =>
          // NOTE: always generate a call as if we're calling a closure. 
          // Since there's no way for us to decide whether we're calling a 
          // closure or not.
          match self.bindings[f_var] {
            Some(maybe_closure) => {
              args.push(Var(maybe_closure))
              // Calling a user level function
              App(Label(f_var), args.map(recrbva))
            }
            None =>
              // Calling an external function
              App(Label(f_var), args.map(recrbva))
          }
        _ => @util.die("Can't invoke call on \{f}")
      }
    Exit => Exit
  }
}
