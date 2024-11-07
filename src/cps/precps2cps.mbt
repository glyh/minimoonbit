// REF: Compiling with Continuations
typealias Cont[C] = (Value) -> Trampo[C]

struct MultiNest[C] {
  nest : ArrayView[Trampo[C]]
  forced : Array[C]
  tmp_stack : Array[(Cps) -> Trampo[Cps]]
  unite : (Array[Cps]) -> Trampo[C]
}

enum Trampo[C] {
  One(P, Cont[C])
  Multi(ArrayView[P], Array[Value], (Array[Value]) -> Trampo[C])
  Nest(MultiNest[C])
  //pub fn CpsEnv::precps2cps(self : CpsEnv, s : P, c : Cont) -> Cps {
  Just(C)
  Then(Trampo[C], (C) -> Trampo[C])
}

// any passed params might not use label, but they should use a var instead
// as they're converted to closures later
fn fix_label_to_var(v : Value) -> Value {
  match v {
    Label(l) => Var(l)
    v => v
  }
}

fn CpsEnv::generate_trampoline(
  self : CpsEnv,
  s : P,
  c : Cont[Cps]
) -> Trampo[Cps] {
  match s {
    Unit => c(Unit)
    Bool(b) => c(Bool(b))
    Int(i) => c(Int(i))
    Double(f) => c(Double(f))
    Let(ty, name, rhs, rest) => {
      fn c1(v : Value) {
        let rest = One(rest, c)
        let orignal_binding = Var::from_precps(name, ty)
        Then(rest, fn(val) { Just(val.replace_var_bind(orignal_binding, v)) })
      }

      One(rhs, c1)
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
        Just(App(Var(k_ref), [returned].map(fix_label_to_var)))
      }

      let trampo_body = One(body, fn_cont)
      let trampo_rest = One(rest, c)
      Nest(
        {
          nest: [trampo_body, trampo_rest][:],
          forced: [],
          tmp_stack: [],
          unite: fn(parts) {
            guard let [body, rest] = parts else {
              _ => @util.die("unreachable")
            }
            Just(Fix(f_ref, new_vars, body, rest))
          },
        },
      )
    }
    Var(ty, v) => c(Var(Var::from_precps(v, ty)))
    Label(ty, v) => c(Label(Var::from_precps(v, ty)))
    Tuple(tup_ty, elements) => {
      fn c1(vs : Array[Value]) {
        let tmp = self.new_tmp(tup_ty)
        Then(c(Var(tmp)), fn(c_tmp) { Just(Tuple(vs, tmp, c_tmp)) })
      }

      Multi(elements[:], [], c1)
    }
    // WARN: Array Creation must come with an external call and thus continuation
    Prim(ret_ty, MakeArray, rands) => {
      let k_ref = self.new_tmp(Fun([ret_ty], Unit))
      let x_ref = self.new_tmp(ret_ty)
      fn c1(args : Array[Value]) {
        // len, elem, kont
        Just(MakeArray(args[0], args[1], Var(k_ref)))
      }

      Nest(
        {
          nest: [c(Var(x_ref)), Multi(rands[:], [], c1)][:],
          forced: [],
          tmp_stack: [],
          unite: fn(parts) {
            guard let [c_x_ref, rands] = parts else {
              _ => @util.die("unreachable")
            }
            Just(Fix(k_ref, [x_ref], c_x_ref, rands))
          },
        },
      )
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
      fn c1(rands : Array[Value]) {
        let tmp = self.new_tmp(ty)
        Then(c(Var(tmp)), fn(c_tmp) { Just(Prim(rator, rands, tmp, c_tmp)) })
      }

      Multi(rands[:], [], c1)
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
            Just(App(f, es))
          }

          Multi(args[:], [], c2)
        }

        Nest(
          {
            nest: [c(Var(x_ref)), One(f, c1)][:],
            forced: [],
            tmp_stack: [],
            unite: fn(parts) {
              guard let [c_x_ref, app_call] = parts else {
                _ => @util.die("unreachable")
              }
              Just(Fix(k_ref, [x_ref], c_x_ref, app_call))
            },
          },
        )
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
            Then(
              c(Var(tmp)),
              fn(c_tmp) { Just(Prim(IntOfFloat, a, tmp, c_tmp)) },
            )
          }

          Multi(args[:], [], c1)
        }
        "float_of_int" => {
          fn c1(a : Array[Value]) {
            let tmp = self.new_tmp(ret_ty)
            Then(
              c(Var(tmp)),
              fn(c_tmp) { Just(Prim(FloatOfInt, a, tmp, c_tmp)) },
            )
          }

          Multi(args[:], [], c1)
        }
        "abs_float" => {
          fn c1(a : Array[Value]) {
            let tmp = self.new_tmp(ret_ty)
            Then(c(Var(tmp)), fn(c_tmp) { Just(Prim(AbsFloat, a, tmp, c_tmp)) })
          }

          Multi(args[:], [], c1)
        }
        _ => generate_app()
      }
    }
    KthTuple(ret_ty, offset, tup) => {
      fn c1(v : Value) {
        let tmp = self.new_tmp(ret_ty)
        Then(c(Var(tmp)), fn(c_tmp) { Just(KthTuple(offset, v, tmp, c_tmp)) })
      }

      One(tup, c1)
    }
    If(ret_ty, Prim(Bool, Eq, [lhs, rhs]), _then, _else) => {
      fn c1(lhs : Value) {
        // To avoid exponential growth in CPS ir, we abstract the outer `c` out.
        fn c2(rhs : Value) {
          let k_ref = self.new_tmp(Fun([ret_ty], Unit))
          let x_ref = self.new_tmp(ret_ty)
          fn c3(branch : Value) {
            Just(App(Var(k_ref), [branch].map(fix_label_to_var)))
          }

          Nest(
            {
              nest: [c(Var(x_ref)), One(_then, c3), One(_else, c3)][:],
              forced: [],
              tmp_stack: [],
              unite: fn(parts) {
                guard let [c_x_ref, _then, _else] = parts else {
                  _ => @util.die("unreachable")
                }
                Just(Fix(k_ref, [x_ref], c_x_ref, IfEq(lhs, rhs, _then, _else)))
              },
            },
          )
        }

        One(rhs, c2)
      }

      One(lhs, c1)
    }
    If(ret_ty, Prim(Bool, Le, [lhs, rhs]), _then, _else) => {
      fn c1(lhs : Value) {
        // To avoid exponential growth in CPS ir, we abstract the outer `c` out.
        fn c2(rhs : Value) {
          let k_ref = self.new_tmp(Fun([ret_ty], Unit))
          let x_ref = self.new_tmp(ret_ty)
          fn c3(branch : Value) {
            Just(App(Var(k_ref), [branch].map(fix_label_to_var)))
          }

          Nest(
            {
              nest: [c(Var(x_ref)), One(_then, c3), One(_else, c3)][:],
              forced: [],
              tmp_stack: [],
              unite: fn(parts) {
                guard let [c_x_ref, _then, _else] = parts else {
                  _ => @util.die("unreachable")
                }
                Just(Fix(k_ref, [x_ref], c_x_ref, IfLe(lhs, rhs, _then, _else)))
              },
            },
          )
        }

        One(rhs, c2)
      }

      One(lhs, c1)
    }
    If(ret_ty, cond, _then, _else) => {
      fn c1(cond : Value) {
        // To avoid exponential growth in CPS ir, we abstract the outer `c` out.
        let k_ref = self.new_tmp(Fun([ret_ty], Unit))
        let x_ref = self.new_tmp(ret_ty)
        fn c2(branch : Value) {
          Just(App(Var(k_ref), [branch].map(fix_label_to_var)))
        }

        Nest(
          {
            nest: [c(Var(x_ref)), One(_then, c2), One(_else, c2)][:],
            forced: [],
            tmp_stack: [],
            unite: fn(parts) {
              guard let [c_x_ref, _then, _else] = parts else {
                _ => @util.die("unreachable")
              }
              Just(Fix(k_ref, [x_ref], c_x_ref, If(cond, _then, _else)))
            },
          },
        )
      }

      One(cond, c1)
    }
  }
}

pub fn cps_conv_trampolined(env : CpsEnv, input : P) -> Cps {
  let root_then_stack = []
  let nest_stack = []
  let then_stacks = [root_then_stack]
  loop One(input, fn { _ => Just(Exit) }) {
    One(p, cont) => continue env.generate_trampoline(p, cont)
    Multi([], collected, cont) => continue cont(collected)
    Multi([p, .. as rest], collected, cont) =>
      continue One(p, fn(v) { Multi(rest, collected + [v], cont) })
    Nest({ nest: [], unite, .. }) => continue unite([])
    Nest({ nest: [hd, .. as nest_rest], tmp_stack, .. } as info) => {
      nest_stack.push({ ..info, nest: nest_rest })
      then_stacks.push(tmp_stack)
      continue hd
    }
    Just(c) => {
      guard let Some(cur_stack) = then_stacks.last() else {
        _ => @util.die("no stack available")
      }
      match cur_stack.pop() {
        None =>
          match nest_stack.pop() {
            None => break c
            Some(nest_info) =>
              match nest_info {
                { nest: [], forced, unite, .. } => {
                  forced.push(c)
                  ignore(then_stacks.pop())
                  continue unite(forced)
                }
                { nest: [hd, .. as rest], forced, .. } => {
                  forced.push(c)
                  nest_stack.push({ ..nest_info, nest: rest })
                  continue hd
                }
              }
          }
        Some(cont) => continue cont(c)
      }
    }
    Then(cur, cont) => {
      guard let Some(cur_stack) = then_stacks.last() else {
        _ => @util.die("no stack available")
      }
      cur_stack.push(cont)
      continue cur
    }
  }
}
