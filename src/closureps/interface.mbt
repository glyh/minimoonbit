pub struct ClosurePS {
  fnblocks : @hashmap.T[Var, FuncDef]
  root : S
  counter : Ref[Int]
  label_to_closure : @hashmap.T[Var, Var]
  func_no_free_vars : @hashset.T[Var]
}

pub fn cps2clops(cnt : Int, s : S) -> ClosurePS {
  let env = CloEnv::new(cnt)
  env.collect_named_fns(s)
  let func_no_free_vars = @hashset.new()
  let root = env.collect_closure(s, func_no_free_vars)
  let counter = env.counter
  let fnblocks = env.fnblocks
  {
    fnblocks,
    root,
    counter,
    label_to_closure: env.pre_clo_gen,
    func_no_free_vars,
  }
}
