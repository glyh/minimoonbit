// REF: Compiling with Continuations
typealias Cont = (Value) -> Cps

fn CpsEnv::precps2cps_list(
  self : CpsEnv,
  a : Array[P],
  c : (Array[Value]) -> Cps
) -> Cps {
  let vals = []
  fn g(a : ArrayView[P]) {
    match a {
      [e, .. as r] =>
        self.precps2cps(
          e,
          fn(v) {
            vals.push(v)
            g(r)
          },
        )
      [] => c(vals)
    }
  }

  g(a[:])
}

// any passed params might not use label, but they should use a var instead
// as they're converted to closures later
fn fix_label_to_var(v : Value) -> Value {
  match v {
    Label(l) => Var(l)
    v => v
  }
}

pub fn CpsEnv::precps2cps(self : CpsEnv, s : P, c : Cont) -> Cps {
  match s {
    Unit => c(Unit)
    Bool(b) => c(Bool(b))
    Int(i) => c(Int(i))
    Double(f) => c(Double(f))
    Let(ty, name, rhs, rest) => {
      fn c1(v : Value) {
        let rest = self.precps2cps(rest, c)
        let orig_binding = Var::from_precps(name, ty)
        rest.replace_var_bind(orig_binding, v)
      }

      self.precps2cps(rhs, c1)
    }
    // NOTE:
    // Any function f of type (a1, a2, a3, .., an) -> r has been transformed into
    // (a1, a2, a3, .., an, (r) -> Unit)
    LetRec(ty, fn_name, fn_args, body, rest) => {
      guard let Fun(arg_tys, ret_ty) = ty else {
        _ => @util.die("Calling a non function")
      }
      () // generate the type for cps converted function
      let new_arg_tys = arg_tys.copy()
      let k_type = T::Fun([ret_ty], Unit)
      new_arg_tys.push(k_type)
      let new_f_type = T::Fun(new_arg_tys, Unit)
      // reference to the continuation
      let new_vars = []
      for var in zip2(fn_args, arg_tys) {
        let var = Var::from_precps(var.0, var.1)
        new_vars.push(var)
      }
      // create the wrapper
      let k_ref = self.new_named("kont_\{fn_name}", k_type)
      new_vars.push(k_ref)
      let f_ref = Var::from_precps(fn_name, new_f_type)
      fn fn_cont(returned : Value) {
        App(Var(k_ref), [returned].map(fix_label_to_var))
      }

      Fix(
        f_ref,
        new_vars,
        self.precps2cps(body, fn_cont),
        self.precps2cps(rest, c),
      )
    }
    Var(ty, v) => c(Var(Var::from_precps(v, ty)))
    Label(ty, v) => c(Label(Var::from_precps(v, ty)))
    Tuple(tup_ty, elements) => {
      fn c1(vs : Array[Value]) {
        let tmp = self.new_tmp(tup_ty)
        Tuple(vs, tmp, c(Var(tmp)))
      }

      self.precps2cps_list(elements, c1)
    }
    // WARN: Array Creation must come with an external call and thus continuation
    Prim(ret_ty, MakeArray, rands) => {
      let k_ref = self.new_tmp(Fun([ret_ty], Unit))
      let x_ref = self.new_tmp(ret_ty)
      fn c1(args : Array[Value]) {
        // len, elem, kont
        MakeArray(args[0], args[1], Var(k_ref))
      }

      Fix(k_ref, [x_ref], c(Var(x_ref)), self.precps2cps_list(rands, c1))
    }
    Prim(ty, rator, rands) => {
      let rator = match rator {
        Not => Not
        Neg(n) => Neg(n)
        Get => Get
        Put => Put
        Math(op, num) => Math(op, num)
        Eq => Eq
        Le => Le
        MakeArray =>
          @util.die(
            "unreachable: precps -> cps, makearray should be dealt with eearlier",
          )
      }
      fn c1(a : Array[Value]) {
        let tmp = self.new_tmp(ty)
        Prim(rator, a, tmp, c(Var(tmp)))
      }

      self.precps2cps_list(rands, c1)
    }
    // (a1, a2, a3, .., an, (r) -> Unit)
    App(ret_ty, f, args) => {
      fn generate_app() {
        let k_ref = self.new_tmp(Fun([ret_ty], Unit))
        let x_ref = self.new_tmp(ret_ty)
        fn c1(f : Value) {
          fn c2(es : Array[Value]) {
            es.push(Var(k_ref))
            es.map_inplace(fix_label_to_var)
            App(f, es)
          }

          self.precps2cps_list(args, c2)
        }

        Fix(k_ref, [x_ref], c(Var(x_ref)), self.precps2cps(f, c1))
      }

      guard let Var(_, var) | Label(_, var) = f else {
        _ => return generate_app()
      }
      guard var.id < 0 else { return generate_app() }
      guard let Some(name) = var.name else { _ => return generate_app() }
      match name {
        "int_of_float" | "truncate" => {
          fn c1(a : Array[Value]) {
            let tmp = self.new_tmp(ret_ty)
            Prim(IntOfFloat, a, tmp, c(Var(tmp)))
          }

          self.precps2cps_list(args, c1)
        }
        "float_of_int" => {
          fn c1(a : Array[Value]) {
            let tmp = self.new_tmp(ret_ty)
            Prim(FloatOfInt, a, tmp, c(Var(tmp)))
          }

          self.precps2cps_list(args, c1)
        }
        "abs_float" => {
          fn c1(a : Array[Value]) {
            let tmp = self.new_tmp(ret_ty)
            Prim(AbsFloat, a, tmp, c(Var(tmp)))
          }

          self.precps2cps_list(args, c1)
        }
        _ => generate_app()
      }
    }
    KthTuple(ret_ty, offset, tup) => {
      fn c1(v : Value) {
        let tmp = self.new_tmp(ret_ty)
        KthTuple(offset, v, tmp, c(Var(tmp)))
      }

      self.precps2cps(tup, c1)
    }
    If(ret_ty, Prim(Bool, Eq, [lhs, rhs]), _then, _else) => {
      fn c1(lhs : Value) -> Cps {
        // To avoid exponential growth in CPS ir, we abstract the outer `c` out.
        fn c2(rhs : Value) -> Cps {
          let k_ref = self.new_tmp(Fun([ret_ty], Unit))
          let x_ref = self.new_tmp(ret_ty)
          fn c3(branch : Value) -> Cps {
            App(Var(k_ref), [branch].map(fix_label_to_var))
          }

          Fix(
            k_ref,
            [x_ref],
            c(Var(x_ref)),
            IfEq(
              lhs,
              rhs,
              self.precps2cps(_then, c3),
              self.precps2cps(_else, c3),
            ),
          )
        }

        self.precps2cps(rhs, c2)
      }

      self.precps2cps(lhs, c1)
    }
    If(ret_ty, Prim(Bool, Le, [lhs, rhs]), _then, _else) => {
      fn c1(lhs : Value) -> Cps {
        // To avoid exponential growth in CPS ir, we abstract the outer `c` out.
        fn c2(rhs : Value) -> Cps {
          let k_ref = self.new_tmp(Fun([ret_ty], Unit))
          let x_ref = self.new_tmp(ret_ty)
          fn c3(branch : Value) -> Cps {
            App(Var(k_ref), [branch].map(fix_label_to_var))
          }

          Fix(
            k_ref,
            [x_ref],
            c(Var(x_ref)),
            IfLe(
              lhs,
              rhs,
              self.precps2cps(_then, c3),
              self.precps2cps(_else, c3),
            ),
          )
        }

        self.precps2cps(rhs, c2)
      }

      self.precps2cps(lhs, c1)
    }
    If(ret_ty, cond, _then, _else) => {
      fn c1(cond : Value) -> Cps {
        // To avoid exponential growth in CPS ir, we abstract the outer `c` out.
        let k_ref = self.new_tmp(Fun([ret_ty], Unit))
        let x_ref = self.new_tmp(ret_ty)
        fn c2(branch : Value) -> Cps {
          App(Var(k_ref), [branch].map(fix_label_to_var))
        }

        Fix(
          k_ref,
          [x_ref],
          c(Var(x_ref)),
          If(cond, self.precps2cps(_then, c2), self.precps2cps(_else, c2)),
        )
      }

      self.precps2cps(cond, c1)
    }
  }
}
