// REF:
// 1. https://course.ccs.neu.edu/cs4410/lec_type-inference_notes.html
// 2. https://cs3110.github.io/textbook/chapters/interp/inference.html
// 3. My own work at https://github.com/glyh/wyah/blob/master/lib/type_inference.ml

// NOTE: 
// Constraint based STLC inference with additonal of type class(for polymorphic operators)

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

fn apply_sub_ty(s : Subst, target : T) -> T {
  fn rec(t : T) {
    apply_sub_ty(s, t)
  }

  match target {
    Unit | Bool | Int | Double | Ptr as t => t
    Fun(args, ret) => Fun(args.map(rec), rec(ret))
    Tuple(tup) => Tuple(tup.map(rec))
    Array(t) => Array(rec(t))
    // NOTE: we rely on the uniqueness of the generated type var to ensure binder
    // is not substituted here 
    Var({ val: Ok(t) } as ref) as ret => {
      ref.val = Ok(rec(t))
      ret
    }
    Var({ val: Err(t) } as ref) as ret => {
      match s[t.0] {
        Some(ty) => ref.val = Ok(ty)
        None => ()
      }
      ret
    }
  }
}

fn apply_sub_sub(s_applied : Subst, s_target : Subst) -> Subst {
  s_target.iter().map(fn { (k, v) => (k, apply_sub_ty(s_applied, v)) })
  |> @immut/sorted_map.from_iter
}

fn apply_sub_cs(s : Subst, eq_cons : Array[Constraint]) -> Array[Constraint] {
  fn rec(ty) {
    apply_sub_ty(s, ty)
  }

  eq_cons.map(fn { (lhs, rhs) => (rec(lhs), rec(rhs)) })
}

fn apply_sub_env(s : Subst, env : TyEnv) -> TyEnv {
  env.iter().map(fn { (k, v) => (k, apply_sub_ty(s, v)) })
  |> @immut/sorted_map.from_iter
}

typealias Resolved = (Ref[Result[T, TVar]], T)

fn unify_one(c : Constraint) -> (Resolved?, Array[Constraint]) {
  let (lhs, rhs) = c
  if lhs == rhs {
    return (None, [])
  }
  match c {
    (Fun(args1, ret1), Fun(args2, ret2)) =>
      if args1.length() != args2.length() {
        @util.die("Unification failure\{c}")
      } else {
        let cs = args1.mapi(fn(i, arg1) { (arg1, args2[i]) })
        cs.push((ret1, ret2))
        (None, cs)
      }
    (Tuple(tup1), Tuple(tup2)) =>
      if tup1.length() != tup2.length() {
        @util.die("Unification failure\{c}")
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
            }
          (Some(({ val: Err((id, clss)) } as ref, tj)), cs_new) => {
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
      match env.lookup(name) {
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
      let (ty_bool, cs_bool) = rec(bool)
      let cs_out : Array[Constraint] = [(ty_bool, T::Bool)]
      cs_out.append(cs_bool)
      (T::Bool, cs_out)
    }
    Array(len, ele) => {
      let (ty_len, cs_len) = rec(len)
      let (ty_ele, cs_ele) = rec(ele)
      let cs_out : Array[Constraint] = [(ty_len, T::Int)]
      cs_out.append(cs_len)
      cs_out.append(cs_ele)
      (Array(ty_ele), cs_out)
    }
    Neg(numeric, kind=_) => {
      let (ty_inner, cs_inner) = rec(numeric)
      let class_var = @types.classed_tvar([Arith])
      cs_inner.push((class_var, ty_inner))
      (ty_inner, cs_inner)
    }
    App(f, params) => {
      let (f_ty, cs_out) = rec(f)
      let ty_params_out : Array[T] = []
      for pi in params.iter() {
        let (ty_i, cs_i) = rec(pi)
        ty_params_out.push(ty_i)
        cs_out.append(cs_i)
      }
      let return_var = @types.dummy_tvar()
      cs_out.push((f_ty, Fun(ty_params_out, return_var)))
      (return_var, cs_out)
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
      let class_var = @types.classed_tvar([Arith])
      cs_out.push((class_var, lhs_ty))
      (lhs_ty, cs_out)
    }
    Eq(lhs, rhs) | LE(lhs, rhs) => {
      let (lhs_ty, cs_out) = rec(lhs)
      let (rhs_ty, cs_rhs) = rec(rhs)
      cs_out.append(cs_rhs)
      cs_out.push((lhs_ty, rhs_ty))
      let class_var = @types.classed_tvar([Order])
      (T::Bool, cs_out)
    }
    Let((bind_id, bind_ty), rhs, rest) => {
      let (rhs_ty, rhs_cons) = inference_constraints(env, rhs)
      let env_new = env.insert(bind_id, bind_ty)
      let (ret_ty, cs_out) = inference_constraints(env_new, rest)
      cs_out.push((rhs_ty, bind_ty))
      cs_out.append(rhs_cons)
      (ret_ty, cs_out)
    }
    LetRec(f, rest) => {
      let fn_ty = T::Fun(f.args.map(fn { (_1, _2) => _2 }), f.name.1)
      fn update_env(env : TyEnv, arg : (Symbol, T)) {
        env.insert(arg.0, arg.1)
      }

      let env_with_fn = update_env(env, (f.name.0, fn_ty))
      let env_with_fn_and_args = f.args.fold(init=env, update_env)
      let (body_ty, lambda_cons) = inference_constraints(
        env_with_fn_and_args,
        f.body,
      )
      lambda_cons.push((body_ty, f.name.1))

      // HACK: let-polymorphism
      //let env_generalized = generalize(lambda_cons, env, f.name.0, fn_ty)
      //let (ret_ty, cs_out) = inference_constraints(env_generalized, rest)
      //(ret_ty, cs_out)

      let env_new = env.insert(f.name.0, fn_ty)
      let (ret_ty, cs_rest) = inference_constraints(env_new, rest)
      cs_rest.append(lambda_cons)
      (ret_ty, cs_rest)
    }
    LetTuple(tup, rhs, rest) => {
      let (rhs_ty, rhs_cons) = inference_constraints(env, rhs)
      fn update_env(env : TyEnv, arg : (Symbol, T)) {
        env.insert(arg.0, arg.1)
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

pub fn infer_type(env : TyEnv, exp : S) -> Unit {
  let (_, cons) = inference_constraints(env, exp)
  unify(cons)
}
