// replace any function vars 
fn CloEnv::rebind_var(self : CloEnv, v : Var) -> Var {
  match self.bindings[v] {
    None => v
    Some(wrapped) => wrapped
  }
}

fn CloEnv::rebind_value(self : CloEnv, v : Value) -> Value {
  match v {
    Var(l) | Label(l) =>
      match self.bindings[l] {
        None => v
        Some(wrapped) => Var(wrapped)
      }
    v => v
  }
}

fn CloEnv::collect_label_closure_val(
  self : CloEnv,
  v : Value,
  output : @hashset.T[Var]
) -> Unit {
  guard let Label(l) = v else { _ => return  }
  if self.named_fns.contains(l) {
    output.insert(l)
  }
}

fn CloEnv::collect_label_closure_vals(
  self : CloEnv,
  vs : Array[Value],
  output : @hashset.T[Var]
) -> Unit {
  vs.each(fn(v) { self.collect_label_closure_val(v, output) })
}

fn CloEnv::collect_label_closure(
  self : CloEnv,
  s : S,
  output : @hashset.T[Var]
) -> Unit {
  match s {
    Prim(_, vals, _, rest) | Tuple(vals, _, rest) => {
      self.collect_label_closure_vals(vals, output)
      self.collect_label_closure(rest, output)
    }
    KthTuple(_, _, _, rest) => self.collect_label_closure(rest, output)
    Fix(fn_name, _, body, rest) => {
      self.collect_label_closure(body, output)
      self.collect_label_closure(rest, output)
      output.remove(fn_name)
    }
    If(_, _then, _else) | IfEq(_, _, _then, _else) | IfLe(_, _, _then, _else) => {
      self.collect_label_closure(_then, output)
      self.collect_label_closure(_else, output)
    }
    App(f, args) => {
      self.collect_label_closure_vals(args, output)
      guard let Label(label) = f else { _ => return  }
      if label.id > 0 {
        output.insert(label)
      }
    }
    MakeArray(_, elem, cont) => {
      self.collect_label_closure_val(elem, output)
      self.collect_label_closure_val(cont, output)
    }
    Exit => ()
  }
}

fn CloEnv::get_closure(
  self : CloEnv,
  fn_name : Var,
  ~closure_ty : T = Unit
) -> Var {
  match self.pre_clo_gen[fn_name] {
    None => {
      let generated = self.new_named("clo_\{fn_name.to_string()}", closure_ty)
      // it's fine to have the type being unit as it'll be fixed later
      self.pre_clo_gen[fn_name] = generated
      generated
    }
    Some(gen) => {
      if closure_ty != Unit {
        gen.ty.val = closure_ty
      }
      gen
    }
  }
}

// collect all closures to top level and fix call convention
// NOTE:
// whenever we store a reference to an external call, we need to wrap it as a closure
fn CloEnv::collect_closure(
  self : CloEnv,
  s : S,
  func_no_free_vars : @hashset.T[Var]
) -> S {
  fn rec(c : S) {
    self.collect_closure(c, func_no_free_vars)
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
        self.add_rebind(bind, bind).collect_closure(rest, func_no_free_vars),
      )
    KthTuple(idx, v, bind, rest) =>
      KthTuple(
        idx,
        v,
        bind,
        self.add_rebind(bind, bind).collect_closure(rest, func_no_free_vars),
      )
    If(_cond, _then, _else) => If(_cond, rec(_then), rec(_else))
    IfEq(lhs, rhs, _then, _else) => IfEq(lhs, rhs, rec(_then), rec(_else))
    IfLe(lhs, rhs, _then, _else) => IfLe(lhs, rhs, rec(_then), rec(_else))
    Prim(op, args, bind, rest) =>
      Prim(
        op,
        args,
        bind,
        self.add_rebind(bind, bind).collect_closure(rest, func_no_free_vars),
      )
    Fix(f, args, body, rest) => {
      // Step 1. Collect free variables to generate free_var & closure
      // NOTE: For unknown functions, we still need to pass them a closure
      // and that param might be free. so we may need to add that as well.
      let free_vars_set = body.free_variables()
      self.collect_label_closure(body, free_vars_set)
      let free_vars = free_vars_set
        .iter()
        .filter(
          fn(v) {
            v != f &&
            not(args.contains(v)) &&
            not(func_no_free_vars.contains(v))
          },
        )
        .map(
          fn(var) {
            if self.named_fns.contains(var) {
              self.get_closure(var)
            } else {
              var
            }
          },
        )
        .collect()
      free_vars.sort()
      let has_free_vars = free_vars.length() != 0
      let free_var_tup_ty = if has_free_vars {
        T::Tuple(free_vars.map(fn(v) { v.ty.val }))
      } else {
        T::Unit
      }
      let free_var_tuple = self.new_named(
        "fv_\{f.to_string()}",
        free_var_tup_ty,
      )
      let closure_ty = T::Tuple([f.ty.val, free_var_tup_ty])
      let closure = self.get_closure(f, ~closure_ty)

      // Step 2. load the free vars to body and recurse collection on it
      let mut body_new = body
      if has_free_vars {
        body_new = free_vars.foldi(
          init=body,
          fn(idx, acc, ele) { KthTuple(idx, Var(free_var_tuple), ele, acc) },
        )
        body_new = KthTuple(1, Var(closure), free_var_tuple, body_new)
      }
      let body_env = args
        .fold(init=self, fn(acc, ele) { acc.add_rebind(ele, ele) })
        .add_rebind(f, closure)
      body_new = body_env.collect_closure(body_new, func_no_free_vars)

      // Step 3. fill this function to fnblocks
      self.fnblocks[f] = { args, free_vars, body: body_new, closure }
      if free_vars.length() == 0 {
        func_no_free_vars.insert(f)
      }

      // Step 4. recurse on rest
      let rest_fixed_inner = self
        .add_rebind(f, closure)
        .collect_closure(rest, func_no_free_vars)
      if has_free_vars {
        let freevars_captured = self.new_named("fvs_cap", free_var_tup_ty)
        Tuple(
          free_vars.map(Value::Var),
          freevars_captured,
          Tuple([Label(f), Var(freevars_captured)], closure, rest_fixed_inner),
        )
      } else {
        rest_fixed_inner
      }
    }
    MakeArray(len, elem, kont) => {
      // pass on closure pointer, kont is generated so it's always a var
      guard let Var(kont_var) = kont else {
        _ => @util.die("generated a label continuation")
      }
      match self.bindings[kont_var] {
        Some(closure) =>
          // NOTE: delay the generation of platform-specific code to later phase
          // This is because the convention for js and riscv is different 
          MakeArray(len, recrbva(elem), recrbva(Var(closure)))
        None =>
          @util.die("undefined continuation \{kont} called after making array")
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
