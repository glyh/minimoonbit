fn collect_externals(cfg : @ssacfg.SsaCfg) -> @hashset.T[Var] {
  let out : @hashset.T[Var] = @hashset.new()
  fn collect_label_var(v : Var) {
    if v.id < 0 {
      out.insert(v)
      //println("adding \{v}, now: \{out}")
    }
  }

  fn collect_label_val(v : Value) {
    match v {
      Label(var) | Var(var) => collect_label_var(var)
      _ => ()
    }
  }

  for item in cfg.blocks {
    //println("start of \{item.0}: \{out}")
    let (_, blk) = item
    for inst in blk.insts {
      match inst {
        MakeTuple(bind, vals) | Prim(bind, _, vals) => {
          collect_label_var(bind)
          vals.each(collect_label_val)
        }
        KthTuple(bind, val, _) => {
          collect_label_var(bind)
          collect_label_val(val)
        }
        Store(var) | Load(var) => collect_label_var(var)
      }
    }
    match blk.last_inst.val {
      Branch(cond, _then, _else) => {
        collect_label_val(cond)
        collect_label_var(_then)
        collect_label_var(_else)
      }
      Call(f, args) => {
        collect_label_val(f)
        args.each(collect_label_val)
      }
      MakeArray(len, elem, kont) => {
        collect_label_val(len)
        collect_label_val(elem)
        collect_label_val(kont)
      }
      Exit => ()
    }
  }
  out
}
