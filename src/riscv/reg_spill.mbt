fn reg_spill_fn(
  cfg : @ssacfg.SsaCfg,
  all_vars_not_preallocated : @hashset.T[Var],
  can_allocate : Array[Var], // vars that can be allocated but not yet
  interference : InterferenceGraph,
  fn_name : Var
) -> Var {
  let spill_candidates = all_vars_not_preallocated
    .iter()
    .filter(fn(var) { not(can_allocate.contains(var)) })
    .collect()
  spill_candidates.sort_by_key(
    fn(var) { interference[var].map(fn(s) { s.size() }).or(0) },
  )
  let spilled_var = spill_candidates.last().unwrap()
  let labels = collect_blocks_of_fn(cfg, fn_name)
  for label in labels {
    let blk = cfg.blocks[label].unwrap()
    cfg.blocks[label] = reg_spill_block(blk, spilled_var)
  }
  spilled_var
}

fn reg_spill_block(blk : @ssacfg.Block, spilled_var : Var) -> @ssacfg.Block {
  let insts_new : Array[@ssacfg.Inst] = []
  fn val_spilled(val : Value) -> Bool {
    match val {
      Var(var) => var == spilled_var
      _ => false
    }
  }

  fn vals_spilled(vals : Array[Value]) -> Bool {
    vals.fold(init=false, fn(acc, ele) { acc || val_spilled(ele) })
  }

  for inst in blk.insts {
    match inst {
      MakeTuple(bind, tup) => {
        if vals_spilled(tup) {
          insts_new.push(Load(spilled_var))
        }
        insts_new.push(inst)
        if bind == spilled_var {
          insts_new.push(Store(spilled_var))
        }
      }
      KthTuple(bind, tup, _) => {
        if val_spilled(tup) {
          insts_new.push(Load(spilled_var))
        }
        insts_new.push(inst)
        if bind == spilled_var {
          insts_new.push(Store(spilled_var))
        }
      }
      Copy(bind, copied) => {
        if val_spilled(copied) {
          insts_new.push(Load(spilled_var))
        }
        insts_new.push(inst)
        if bind == spilled_var {
          insts_new.push(Store(spilled_var))
        }
      }
      Prim(bind, _, args) => {
        if vals_spilled(args) {
          insts_new.push(Load(spilled_var))
        }
        insts_new.push(inst)
        if bind == spilled_var {
          insts_new.push(Store(spilled_var))
        }
      }
      Load(_) | Store(_) => insts_new.push(inst)
    }
  }
  let load_before_exit = match blk.last_inst.val {
    Branch(cond, _, _) => val_spilled(cond)
    BranchEq(lhs, rhs, _, _) | BranchLe(lhs, rhs, _, _) =>
      val_spilled(lhs) || val_spilled(rhs)
    Call(f, args) => val_spilled(f) || vals_spilled(args)
    MakeArray(len, elem, kont) =>
      val_spilled(len) || val_spilled(elem) || val_spilled(kont)
    Exit => false
  }
  if load_before_exit {
    insts_new.push(Load(spilled_var))
  }
  { ..blk, insts: insts_new }
}
