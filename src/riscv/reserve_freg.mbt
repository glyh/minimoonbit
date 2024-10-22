fn reserve_freg(cfg : @ssacfg.SsaCfg) -> @ssacfg.SsaCfg {
  for item in cfg.blocks {
    let (_, block) = item
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
  cfg
}
