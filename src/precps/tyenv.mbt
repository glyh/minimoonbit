pub struct TyEnv {
  bindings : @immut/hashmap.T[Either[Int, String], (Var, T)]
  counter : Ref[Int]
}

fn to_bind_key(item : (String, T)) -> (Either[Int, String], (Var, T)) {
  let ext_name = item.0
  let ty = item.1
  (Right(ext_name), (var_of_external(ext_name), ty))
}

fn find_bind_key(v : Var) -> Either[Int, String] {
  match v.name {
    None => Left(v.id)
    Some(s) => Right(s)
  }
}

pub fn TyEnv::new(externals : @immut/hashmap.T[String, T]) -> TyEnv {
  let externals = externals.iter().map(to_bind_key)
    |> @immut/hashmap.from_iter()
  { bindings: externals, counter: { val: 0 } }
}

pub fn TyEnv::gen_tmp(self : TyEnv) -> Var {
  self.counter.val = self.counter.val + 1
  { id: self.counter.val, name: None }
}

pub fn TyEnv::add(self : TyEnv, name : String, ty : T) -> (Var, TyEnv) {
  self.counter.val = self.counter.val + 1
  let to_bind = { id: self.counter.val, name: Some(name) }
  let bindings = self.bindings.add(find_bind_key(to_bind), (to_bind, ty))
  (to_bind, { ..self, bindings, })
}

pub fn TyEnv::add_many(self : TyEnv, args : Iter[(String, T)]) -> TyEnv {
  args.fold(
    init=self,
    fn(acc, ele) {
      let (_, env_new) = acc.add(ele.0, ele.1)
      env_new
    },
  )
}

pub fn TyEnv::find(self : TyEnv, name : String) -> PreCps? {
  match self.bindings[Right(name)] {
    Some(item) => Some(Var(item.1, item.0))
    None =>
      match self.bindings[Right(name)] {
        None => None
        Some(item) => Some(Var(item.1, item.0))
      }
  }
}
