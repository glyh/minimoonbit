// NOTE:
// After marking each function to be closure or not, we need another phase to 
// decide whether or not to wrap calls to functions to fix the signature at call
// site. This, however, need us to know corre

pub type Label String derive(Show, Hash, Eq)

struct CloEnv {
  // NOTE: we only want fundefs to contaminate as it's at top-level
  fundefs : @hashmap.T[Label, FuncDef]
  externals : @hashmap.T[String, LowType]
  tyenv : @immut/hashmap.T[Name, Either[LowType, Label]]
  mut counter : Int
} derive(Show)

fn CloEnv::new(externals : Map[String, Type], knf : K) -> CloEnv {
  let externals = externals
    .iter()
    .map(fn { (name, ty) => ("minimbt_" + name, LowType::of_ty(ty)) })
    |> @hashmap.from_iter()
  let counter = @knf.knf_max_counter(knf) + 1
  { fundefs: @hashmap.new(), tyenv: @immut/hashmap.new(), counter, externals }
}

fn CloEnv::insert_ty(self : CloEnv, name : Name, ty : LowType) -> CloEnv {
  { ..self, tyenv: self.tyenv.add(name, Left(ty)) }
}

fn CloEnv::insert_fun(
  self : CloEnv,
  name : Name,
  f : FuncDef
) -> (Label, CloEnv) {
  let label = "\{name}_\{self.counter}"
  self.fundefs[label] = { ..f, name: label, old_name: name }
  (label, { ..self, tyenv: self.tyenv.add(name, Right(label)) })
}

fn CloEnv::inside_fun(self : CloEnv, f : FuncDef) -> CloEnv {
  let tyenv = (f.args + f.formal_free_vars).fold(
    init=self.tyenv.add(f.old_name, Right(f.name)),
    fn(acc, arg) {
      let (id, ty) = arg
      acc.add(id, Left(ty))
    },
  )
  { ..self, tyenv, }
}

fn CloEnv::find_ty(self : CloEnv, n : Name) -> LowType? {
  match self.tyenv[n] {
    None => None
    Some(Left(ty)) => Some(ty)
    Some(Right(label)) =>
      match self.fundefs[label] {
        None => None
        Some(func) => Some(func.ty)
      }
  }
}

fn CloEnv::collect_closure(self : CloEnv, k : K) -> Expr {
  match k {
    Unit => Unit
    Int(i) => Int(i)
    Double(f) => Double(f)
    Var(name) =>
      match self.tyenv.find(name) {
        Some(_) => Var(name)
        None =>
          match self.externals[name.to_string()] {
            None =>
              @util.die("undefined variable \{name} in \{k} under \{self}")
            Some(ty) =>
              MakeClosure(
                (name, ty),
                { name: name.to_string(), actual_free_vars: [] },
                Var(name),
              )
          }
      }
    Tuple(names) => MakeTuple(names)
    Neg(name) => Neg(name)
    Apply(f, args) => CallClosure(f, args)
    Get(arr, idx) => ArrayGet(arr, idx)
    IfEq(lhs, rhs, _then, _else) => {
      let _then = self.collect_closure(_then)
      let _else = self.collect_closure(_else)
      IfEq(lhs, rhs, _then, _else)
    }
    IfLe(lhs, rhs, _then, _else) => {
      let _then = self.collect_closure(_then)
      let _else = self.collect_closure(_else)
      IfLe(lhs, rhs, _then, _else)
    }
    Add(lhs, rhs) => Add(lhs, rhs)
    Sub(lhs, rhs) => Sub(lhs, rhs)
    Mul(lhs, rhs) => Sub(lhs, rhs)
    Div(lhs, rhs) => Sub(lhs, rhs)
    FNeg(e) => FNeg(e)
    FAdd(lhs, rhs) => FAdd(lhs, rhs)
    FSub(lhs, rhs) => FSub(lhs, rhs)
    FMul(lhs, rhs) => FMul(lhs, rhs)
    FDiv(lhs, rhs) => FDiv(lhs, rhs)
    Let((name, ty), rhs, rest) => {
      let rhs = self.collect_closure(rhs)
      let ty = LowType::of_ty_gen(ty)
      let rest = self.insert_ty(name, ty).collect_closure(rest)
      Let((name, ty), rhs, rest)
    }
    LetRec(f, rest) => {
      let free_variables = f.free_variables()
      let is_closure = not(free_variables.is_empty())
      let args = f.args.map(fn { (arg, ty) => (arg, LowType::of_ty_gen(ty)) })
      let arg_names = args.map(fn { (arg, _) => arg })
      let formal_free_vars = free_variables
        .iter()
        .filter(fn { fv => not(arg_names.contains(fv)) })
        .map(fn { fv => (fv, self.find_ty(fv).unwrap()) })
        .collect()
      let ty = LowType::of_ty(f.ty)
      if is_closure {
        ty.mark_as_closure()
      }
      let func : FuncDef = {
        name: "dummy",
        old_name: f.name,
        is_closure,
        ty,
        args,
        formal_free_vars,
        body: Unit,
      }
      let body = self.inside_fun(func).collect_closure(f.body)
      func.body = body
      let (label, env_new) = self.insert_fun(f.name, func)
      let rest = env_new.collect_closure(rest)
      MakeClosure(
        (f.name, ty),
        { name: label, actual_free_vars: free_variables.iter().collect() },
        rest,
      )
    }
    LetTuple(tup, rhs, rest) => {
      guard let Some(Tuple(tup_ty)) = self.find_ty(rhs) else {
        _ =>
          @util.die(
            "Inferred \{rhs} to be non tuple during closure conv under \{self}",
          )
      }
      if tup_ty.length() != tup.length() {
        @util.die(
          "Tuple length mismatch when doing closure conv for \{tup} = \{rhs}",
        )
      }
      let tup_lowed = tup.mapi(
        fn(idx, id_ty) {
          let (id, _) = id_ty
          let ty = tup_ty[idx]
          (id, ty)
        },
      )
      let env_new = tup_lowed.fold(
        init=self,
        fn(env, ele) {
          let (name, ty) = ele
          env.insert_ty(name, ty)
        },
      )
      let rest_lowed = env_new.collect_closure(rest)
      LetTuple(tup_lowed, rhs, rest_lowed)
    }
    Put(arr, idx, rhs) => ArrayPut(arr, idx, rhs)
    ExternalArray(name) => ExternalArray(name.to_string())
    ExternalFunctionApplication(id, params) => CallDirect(id, params)
  }
}
