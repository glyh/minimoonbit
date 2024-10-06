struct CloEnv {
  // NOTE: fundef's type's arg's length should be one more than FuncDef.args, as
  // we use the last slot for closure.
  // and the types is recursive, BTW
  fnblocks : @hashmap.T[Var, FuncDef]
  counter : Ref[Int]
  bindings : @immut/hashmap.T[Var, Var]
}

fn CloEnv::add_rebind(self : CloEnv, name : Var, cb : Var) -> CloEnv {
  { ..self, bindings: self.bindings.add(name, cb) }
}

fn CloEnv::new(counter : Int) -> CloEnv {
  let counter = { val: counter }
  { fnblocks: @hashmap.new(), counter, bindings: @immut/hashmap.new() }
}

fn CloEnv::new_tmp(self : CloEnv, t : T) -> Var {
  self.counter.val = self.counter.val + 1
  { name: { val: None }, id: self.counter.val, ty: t }
}

// NOTE: no worry of repeated names generated as all vars are marked by an uid
fn CloEnv::new_named(self : CloEnv, name : String, t : T) -> Var {
  self.counter.val = self.counter.val + 1
  { name: { val: Some(name) }, id: self.counter.val, ty: t }
}
