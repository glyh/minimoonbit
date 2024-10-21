pub fn clops2ssacfg(clops : @closureps.ClosurePS) -> SsaCfg {
  let cfg = {
    blocks: @hashmap.new(),
    fn_args: @hashmap.new(),
    spilled: @hashmap.new(),
    counter: clops.counter.val,
  }
  let entry_label : @cps.Var = {
    name: { val: Some("main") },
    id: 0,
    ty: { val: T::Unit },
  }
  let entry_block = cfg.new_entry(entry_label)
  cfg.blocks[entry_label] = entry_block
  cfg.cps2block(clops.root, entry_block, entry_label)
  for item in clops.fnblocks {
    let (fn_name, func) = item
    let fn_block = cfg.new_fn_block(fn_name, func)
    cfg.blocks[fn_name] = fn_block
    cfg.cps2block(func.body, fn_block, fn_name)
  }
  cfg
}
