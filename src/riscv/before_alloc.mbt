// NOTE: reserve a reg for float comparison when both operand are tmps
fn reserve_fregs(cfg : @ssacfg.SsaCfg, block : @ssacfg.Block) -> Unit {
  let insts_backup = block.insts
  let insts = []
  for inst in insts_backup {
    match inst {
      Prim(bind, Eq | Le as op, [lhs, rhs]) as inst =>
        match lhs.get_type() {
          Double => {
            let lhs_is_imm = match lhs {
              Double(_) => true
              _ => false
            }
            let rhs_is_imm = match rhs {
              Double(_) => true
              _ => false
            }
            if lhs_is_imm && rhs_is_imm {
              let tmp = cfg.new_named("tmp", ty=Double)
              insts.push(@ssacfg.Inst::Copy(tmp, lhs))
              insts.push(Prim(bind, op, [Var(tmp), rhs]))
            } else {
              insts.push(Prim(bind, op, [lhs, rhs]))
            }
          }
          _ => insts.push(inst)
        }
      inst => insts.push(inst)
    }
  }
  block.insts = insts
}

// NOTE: load constant closure into scope whenever it's needed
fn ensure_closure_load(
  all_constant_closures : @hashset.T[Var],
  block : @ssacfg.Block
) -> Unit {
  let insts_backup = block.insts
  let insts : Array[@ssacfg.Inst] = []
  let defined_constant_closure : @hashset.T[Var] = @hashset.new()
  fn check_constant_closure_val(v : Value) {
    match v {
      Var(v) =>
        if all_constant_closures.contains(v) &&
          not(defined_constant_closure.contains(v)) {
          insts.push(LoadAddr(v, v.to_string()))
          defined_constant_closure.insert(v)
        }
      _ => ()
    }
  }

  for inst in insts_backup {
    match inst {
      MakeTuple(bind, vals) => {
        vals.each(check_constant_closure_val)
        defined_constant_closure.insert(bind)
      }
      KthTuple(bind, tup, _) => {
        check_constant_closure_val(tup)
        defined_constant_closure.insert(bind)
      }
      Prim(bind, _, args) => {
        args.each(check_constant_closure_val)
        defined_constant_closure.insert(bind)
      }
      Store(_) | Load(_) =>
        @util.die("unreachable: load/store occurs before allocation")
      Copy(bind, copied) => {
        check_constant_closure_val(copied)
        defined_constant_closure.insert(bind)
      }
      LoadAddr(bind, _) => defined_constant_closure.insert(bind)
    }
    insts.push(inst)
  }
  block.insts = insts
}

fn before_alloc(cfg : @ssacfg.SsaCfg) -> @ssacfg.SsaCfg {
  let all_constant_closures = cfg.label_to_closure
    .iter()
    .filter(fn { (label, _) => cfg.func_no_free_vars.contains(label) })
    .map(fn { (_, v) => v })
    |> @hashset.from_iter()
  for item in cfg.blocks {
    let (_, block) = item
    reserve_fregs(cfg, block)
    ensure_closure_load(all_constant_closures, block)
  }
  cfg
}
