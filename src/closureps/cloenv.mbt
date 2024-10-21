struct CloEnv {
  // NOTE: fundef's type's arg's length should be one more than FuncDef.args, as
  // we use the last slot for closure.
  // and the types is recursive, BTW
  fnblocks : @hashmap.T[Var, FuncDef]
  counter : Ref[Int]
  bindings : @immut/hashmap.T[Var, Var]
  // NOTE: when collecting free vars of function body, 
  // we may have to issue the collecter there's actually a closure pointer passed
  // even if closure conversion is not done
  pre_clo_gen : @hashmap.T[Var, Var]
  named_fns : @hashset.T[Var]
}

fn CloEnv::collect_named_fns(self : CloEnv, c : @cps.Cps) -> Unit {
  match c {
    Tuple(_, _, c) | KthTuple(_, _, _, c) | Prim(_, _, _, c) =>
      self.collect_named_fns(c)
    Fix(f, _, body, rest) => {
      self.named_fns.insert(f)
      self.collect_named_fns(body)
      self.collect_named_fns(rest)
    }
    If(_, _then, _else) => {
      self.collect_named_fns(_then)
      self.collect_named_fns(_else)
    }
    App(_) | MakeArray(_) | Exit => ()
  }
}

fn CloEnv::add_rebind(self : CloEnv, name : Var, cb : Var) -> CloEnv {
  { ..self, bindings: self.bindings.add(name, cb) }
}

fn CloEnv::new(counter : Int) -> CloEnv {
  let counter = { val: counter }
  {
    fnblocks: @hashmap.new(),
    counter,
    bindings: @immut/hashmap.new(),
    pre_clo_gen: @hashmap.new(),
    named_fns: @hashset.new(),
  }
}

fn CloEnv::new_tmp(self : CloEnv, t : T) -> Var {
  self.counter.val = self.counter.val + 1
  { name: { val: None }, id: self.counter.val, ty: { val: t } }
}

// NOTE: no worry of repeated names generated as all vars are marked by an uid
fn CloEnv::new_named(self : CloEnv, name : String, t : T) -> Var {
  self.counter.val = self.counter.val + 1
  { name: { val: Some(name) }, id: self.counter.val, ty: { val: t } }
}
