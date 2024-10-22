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
  match block.last_inst.val {
    BranchEq(lhs, rhs, _then, _else) => {
      let lhs_is_imm = match lhs {
        Var(_) => false
        _ => true
      }
      let rhs_is_imm = match rhs {
        Var(_) => false
        _ => true
      }
      if lhs_is_imm && rhs_is_imm {
        let tmp = cfg.new_named("tmp", ty=lhs.get_type())
        insts.push(@ssacfg.Inst::Copy(tmp, lhs))
        block.last_inst.val = BranchEq(Var(tmp), rhs, _then, _else)
      }
    }
    BranchLe(lhs, rhs, _then, _else) => {
      let lhs_is_imm = match lhs {
        Var(_) => false
        _ => true
      }
      let rhs_is_imm = match rhs {
        Var(_) => false
        _ => true
      }
      if lhs_is_imm && rhs_is_imm {
        let tmp = cfg.new_named("tmp", ty=lhs.get_type())
        insts.push(@ssacfg.Inst::Copy(tmp, lhs))
        block.last_inst.val = BranchLe(Var(tmp), rhs, _then, _else)
      }
    }
    _ => ()
  }
  block.insts = insts
}

// NOTE: replace constant closure with labels
fn freeze_closure(
  all_constant_closures : @hashset.T[Var],
  block : @ssacfg.Block
) -> Unit {
  let insts_backup = block.insts
  let insts : Array[@ssacfg.Inst] = []
  fn fix_val(v : Value) -> Value {
    match v {
      Var(v) =>
        if all_constant_closures.contains(v) {
          Label(v)
        } else {
          Var(v)
        }
      _ => v
    }
  }

  for inst in insts_backup {
    let inst : @ssacfg.Inst = match inst {
      MakeTuple(bind, vals) => MakeTuple(bind, vals.map(fix_val))
      KthTuple(bind, tup, k) => KthTuple(bind, fix_val(tup), k)
      Prim(bind, op, args) => Prim(bind, op, args.map(fix_val))
      Store(_) | Load(_) =>
        @util.die("unreachable: load/store occurs before allocation")
      Copy(bind, copied) => Copy(bind, fix_val(copied))
    }
    insts.push(inst)
  }
  block.insts = insts
  block.last_inst.val = match block.last_inst.val {
    Branch(cond, _then, _else) => Branch(cond, _then, _else)
    BranchEq(lhs, rhs, _then, _else) => BranchEq(lhs, rhs, _then, _else)
    BranchLe(lhs, rhs, _then, _else) => BranchLe(lhs, rhs, _then, _else)
    Call(f, args) => Call(fix_val(f), args.map(fix_val))
    MakeArray(len, elem, kont) => MakeArray(len, fix_val(elem), fix_val(kont))
    Exit => Exit
  }
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
    freeze_closure(all_constant_closures, block)
  }
  cfg
}
