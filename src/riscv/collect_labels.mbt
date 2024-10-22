struct ExternalLabels {
  returns_i : @hashset.T[Var]
  returns_f : @hashset.T[Var]
}

fn collect_externals(cfg : @ssacfg.SsaCfg) -> ExternalLabels {
  let out = { returns_i: @hashset.new(), returns_f: @hashset.new() }
  fn collect_label_var(v : Var) {
    if v.id < 0 {
      guard let Fun(_, ret) = v.ty.val else {
        _ => @util.die("unexpected non function external: \{v}")
      }
      match ret {
        Double => out.returns_f.insert(v)
        _ => out.returns_i.insert(v)
      }
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
        KthTuple(bind, val, _) | Copy(bind, val) => {
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
