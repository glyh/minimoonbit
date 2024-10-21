pub struct ClosurePS {
  fnblocks : @hashmap.T[Var, FuncDef]
  root : S
  counter : Ref[Int]
}

pub fn cps2clops(cnt : Int, s : S) -> ClosurePS {
  let env = CloEnv::new(cnt)
  env.collect_named_fns(s)
  let root = env.collect_closure(s)
  let counter = env.counter
  let fnblocks = env.fnblocks
  { fnblocks, root, counter }
}
