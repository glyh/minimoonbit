// REF:
// 1. https://course.ccs.neu.edu/cs4410/lec_type-inference_notes.html
// 2. https://cs3110.github.io/textbook/chapters/interp/inference.html
// 3. My own work at https://github.com/glyh/wyah/blob/master/lib/type_inference.ml

// NOTE: 
// - Constraint based STLC inference with typeclass(for polymorphic operators)
// - We don't actually need substitution here as it's done by directly writing into the pointers

fn free_variables(t : T) -> TVarSet {
  match t {
    Unit | Bool | Int | Double | Ptr => TVarSet::new()
    Fun(args, ret) =>
      args.iter().map(free_variables).fold(init=TVarSet::new(), TVarSet::op_add) +
      free_variables(ret)
    Tuple(tup) =>
      tup.iter().map(free_variables).fold(init=TVarSet::new(), TVarSet::op_add)
    Array(t) => free_variables(t)
    Var({ val: Ok(ty) }) => free_variables(ty)
    Var({ val: Err(var) }) => TVarSet::single(var)
  }
}

typealias Resolved = (Ref[Result[T, TVar]], T)

fn unify_one(c : Constraint) -> (Resolved?, Array[Constraint]) {
  let lhs = c.0.normalize()
  let rhs = c.1.normalize()
  if lhs == rhs {
    return (None, [])
  }
  match (lhs, rhs) {
    (Fun(args1, ret1), Fun(args2, ret2)) =>
      if args1.length() != args2.length() {
        @util.die("Args number mismatch for function unification \{c}")
      } else {
        let cs = args1.mapi(fn(i, arg1) { (arg1, args2[i]) })
        cs.push((ret1, ret2))
        (None, cs)
      }
    (Tuple(tup1), Tuple(tup2)) =>
      if tup1.length() != tup2.length() {
        @util.die("Tuple size mismatch for unification \{c}")
      } else {
        let cs = tup1.mapi(fn(i, t1) { (t1, tup2[i]) })
        (None, cs)
      }
    (Array(l), Array(r)) => (None, [(l, r)])
    (Var({ val: Err(tvar) } as ref), ty) | (ty, Var({ val: Err(tvar) } as ref)) =>
      if free_variables(ty).contains(tvar.0) {
        @util.die("Recursive unification \{c}")
      } else {
        (Some((ref, ty)), [])
      }
    _ => @util.die("Unification failure \{c}")
  }
}

fn unify(cs : Array[Constraint]) -> Unit {
  while true {
    match cs.pop() {
      None => break
      Some(c) =>
        match unify_one(c) {
          (None, cs_new) => cs.append(cs_new)
          (Some(({ val: Ok(ti) }, tj)), cs_new) =>
            if ti != tj {
              @util.die("Can't unify between \{ti} and \{tj}")
            } else {
              cs.append(cs_new)
            }
          (
            Some(
              (
                { val: Err((_, clss1)) } as ref1,
                Var({ val: Err((id2, clss2)) } as ref2),
              )
            ), // both var unknown, we just need to union their clsses
            cs_new,
          ) => {
            ref2.val = Err((id2, clss1.union(clss2)))
            ref1.val = Ok(Var(ref2))
            cs.append(cs_new)
          }
          (Some(({ val: Err((_, clss)) } as ref, tj)), cs_new) => {
            check_classes(tj, clss)
            ref.val = Ok(tj)
            //NOTE: no need to apply the subsitution into cs_new here, as we're using references
            cs.append(cs_new)
          }
        }
    }
  }
}

typealias S = @types.Syntax

fn inference_constraints(env : TyEnv, s : S) -> (T, Array[Constraint]) {
  fn rec(exp : S) {
    inference_constraints(env, exp)
  }

  match s {
    Unit => (T::Unit, [])
    Bool(_) => (T::Bool, [])
    Int(_) => (T::Int, [])
    Double(_) => (T::Double, [])
    Var(name) =>
      match env[name] {
        None => @util.die("Undefined Variable: \{name}")
        Some(ty) => (ty, [])
      }
    Tuple(tup) => {
      let ty_out_arr : Array[T] = []
      let cs_out : Array[Constraint] = []
      for ti in tup.iter() {
        let (ty_i, cs_i) = rec(ti)
        ty_out_arr.push(ty_i)
        cs_out.append(cs_i)
      }
      (Tuple(ty_out_arr), cs_out)
    }
    Not(bool) => {
      let (ty_bool, cs_out) = rec(bool)
      cs_out.push((ty_bool, T::Bool))
      (T::Bool, cs_out)
    }
    Array(len, ele) => {
      let (ty_len, cs_out) = rec(len)
      let (ty_ele, cs_ele) = rec(ele)
      cs_out.append(cs_ele)
      cs_out.push((ty_len, T::Int))
      (Array(ty_ele), cs_out)
    }
    Neg(numeric, ~kind) => {
      let (ty_inner, cs_out) = rec(numeric)
      cs_out.push((kind, ty_inner))
      (ty_inner, cs_out)
    }
    App(f, params) => {
      let (f_ty, cs_out) = rec(f)
      let ty_params : Array[T] = []
      for pi in params.iter() {
        let (ty_i, cs_i) = rec(pi)
        ty_params.push(ty_i)
        cs_out.append(cs_i)
      }
      let ty_ret = @types.dummy_tvar()
      cs_out.push((f_ty, Fun(ty_params, ty_ret)))
      (ty_ret, cs_out)
    }
    Get(arr, index) => {
      let (arr_ty, cs_out) = rec(arr)
      let (index_ty, cs_index) = rec(index)
      cs_out.append(cs_index)
      let return_var = @types.dummy_tvar()
      cs_out.push((Array(return_var), arr_ty))
      cs_out.push((T::Int, index_ty))
      (return_var, cs_out)
    }
    If(cond, then_clause, else_clause) => {
      let (cond_ty, cs_out) = rec(cond)
      let (then_ty, cs_then) = rec(then_clause)
      let (else_ty, cs_else) = rec(else_clause)
      cs_out.append(cs_then)
      cs_out.append(cs_else)
      cs_out.push((cond_ty, T::Bool))
      cs_out.push((then_ty, else_ty))
      (then_ty, cs_out)
    }
    Prim(lhs, rhs, Add | Sub | Mul | Div, ~kind) => {
      let (lhs_ty, cs_out) = rec(lhs)
      let (rhs_ty, cs_rhs) = rec(rhs)
      cs_out.append(cs_rhs)
      cs_out.push((lhs_ty, rhs_ty))
      cs_out.push((kind, lhs_ty))
      (lhs_ty, cs_out)
    }
    Eq(lhs, rhs) | LE(lhs, rhs) => {
      let (lhs_ty, cs_out) = rec(lhs)
      let (rhs_ty, cs_rhs) = rec(rhs)
      cs_out.append(cs_rhs)
      cs_out.push((lhs_ty, rhs_ty))
      let class_var = @types.classed_tvar([Order])
      cs_out.push((class_var, lhs_ty))
      (T::Bool, cs_out)
    }
    Let((bind_id, bind_ty), rhs, rest) => {
      let (rhs_ty, rhs_cons) = inference_constraints(env, rhs)
      let env_new = env.add(bind_id, bind_ty)
      let (ret_ty, cs_out) = inference_constraints(env_new, rest)
      cs_out.push((rhs_ty, bind_ty))
      cs_out.append(rhs_cons)
      (ret_ty, cs_out)
    }
    LetRec(f, rest) => {
      fn update_env(env : TyEnv, arg : (Symbol, T)) {
        env.add(arg.0, arg.1)
      }

      let env_with_fn = update_env(env, f.name)
      let env_with_fn_and_args = f.args.fold(init=env_with_fn, update_env)
      let (body_ty, lambda_cons) = inference_constraints(
        env_with_fn_and_args,
        f.body,
      )
      lambda_cons.push(
        (T::Fun(f.args.map(fn { (_1, ty) => ty }), body_ty), f.name.1),
      )

      // HACK: let-polymorphism
      //let env_generalized = generalize(lambda_cons, env, f.name.0, fn_ty)
      //let (ret_ty, cs_out) = inference_constraints(env_generalized, rest)
      //(ret_ty, cs_out)

      let env_new = env.add(f.name.0, f.name.1)
      let (ret_ty, cs_out) = inference_constraints(env_new, rest)
      cs_out.append(lambda_cons)
      (ret_ty, cs_out)
    }
    LetTuple(tup, rhs, rest) => {
      let (rhs_ty, rhs_cons) = inference_constraints(env, rhs)
      fn update_env(env : TyEnv, arg : (Symbol, T)) {
        env.add(arg.0, arg.1)
      }

      let env_new = tup.fold(init=env, update_env)
      let (ret_ty, cs_out) = inference_constraints(env_new, rest)
      cs_out.append(rhs_cons)
      cs_out.push((rhs_ty, T::Tuple(tup.map(fn { (_1, _2) => _2 }))))
      (ret_ty, cs_out)
    }
    Put(lhs, rhs, rest) => {
      let (lhs_ty, cs_out) = inference_constraints(env, lhs)
      let (rhs_ty, rhs_cons) = inference_constraints(env, rhs)
      let (rest_ty, rest_cons) = inference_constraints(env, rest)
      cs_out.append(rhs_cons)
      cs_out.append(rest_cons)
      cs_out.push((lhs_ty, rhs_ty))
      (rest_ty, cs_out)
    }
  }
}

// 1. ensures every type annotated on the tree is inferred
// 2. normalize any types
fn post_infer_check(exp : S) -> S {
  let rec = post_infer_check
  match exp {
    Unit | Bool(_) | Int(_) | Double(_) | Var(_) => exp
    Tuple(tup) => Tuple(tup.map(rec))
    Not(e) => rec(e)
    Array(len, ele) => Array(rec(len), rec(ele))
    Neg(e, ~kind) => {
      let k_norm = kind.normalize()
      if not(k_norm.is_known()) {
        @util.die("Unknown type for \{exp}")
      }
      Neg(rec(e), kind=k_norm)
    }
    App(f, args) => App(rec(f), args.map(rec))
    Get(arr, idx) => Get(rec(arr), rec(idx))
    If(_test, _then, _else) => If(rec(_test), rec(_then), rec(_else))
    Prim(lhs, rhs, op, ~kind) => {
      let k_norm = kind.normalize()
      if not(k_norm.is_known()) {
        @util.die("Unknown type for \{exp}")
      }
      Prim(rec(lhs), rec(rhs), op, kind=k_norm)
    }
    Eq(lhs, rhs) => Eq(rec(lhs), rec(rhs))
    LE(lhs, rhs) => LE(rec(lhs), rec(rhs))
    Let((name, ty), rhs, rest) => {
      let ty_norm = ty.normalize()
      if not(ty_norm.is_known()) {
        @util.die("Unknown type for \{exp}")
      }
      Let((name, ty_norm), rec(rhs), rec(rest))
    }
    LetRec({ name: (fn_name, fn_ty), args, body }, rest) => {
      let fn_ty_norm = fn_ty.normalize()
      if not(fn_ty_norm.is_known()) {
        @util.die("Unknown type for \{exp}")
      }
      let args_norm = args.map(
        fn {
          (name, ty) => {
            let arg_ty_norm = ty.normalize()
            if not(arg_ty_norm.is_known()) {
              @util.die("Unknown type for \{exp}")
            }
            (name, arg_ty_norm)
          }
        },
      )
      LetRec(
        { name: (fn_name, fn_ty_norm), args: args_norm, body: rec(body) },
        rec(rest),
      )
    }
    LetTuple(tup, rhs, rest) => {
      let tup_norm = tup.map(
        fn {
          (name, ty) => {
            let ty_norm = ty.normalize()
            if not(ty_norm.is_known()) {
              @util.die("Unknown type for \{exp}")
            }
            (name, ty_norm)
          }
        },
      )
      LetTuple(tup_norm, rec(rhs), rec(rest))
    }
    Put(lhs, rhs, rest) => Put(rec(lhs), rec(rhs), rec(rest))
  }
}

pub fn infer_type(exp : S) -> S {
  let (_, cons) = inference_constraints(@types.externals, exp)
  unify(cons)
  post_infer_check(exp)
}
