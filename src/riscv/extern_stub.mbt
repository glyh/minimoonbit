fn collect_externals(cfg : @ssacfg.SsaCfg) -> @hashset.T[Var] {
  let out = @hashset.new()
  fn collect_label_var(v : Var) {
    if v.id < 0 {
      guard let Fun(_) = v.ty.val else {
        _ => @util.die("unexpected non function external: \{v}")
      }
      out.insert(v)
    }
  }

  fn collect_label_val(v : Value) {
    match v {
      Label(var) | Var(var) => collect_label_var(var)
      _ => ()
    }
  }

  for item in cfg.blocks {
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
      BranchEq(lhs, rhs, _then, _else) | BranchLe(lhs, rhs, _then, _else) => {
        collect_label_val(lhs)
        collect_label_val(rhs)
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
        // NOTE: make array itself doesn't count because we generate direct call
        // to unwrapped functions
      }
      Exit => ()
    }
  }
  out
}

fn generate_stub(cfg : @ssacfg.SsaCfg) -> AssemblyFunction {
  // generate stub for CPS / C-Calling convention interop
  let stub_label = cfg.new_named("c_stub").to_string()
  let body : Array[RvAsm] = []
  let externals = collect_externals(cfg)
  // generate stubs for function returning ints
  let kont_arg_at = @hashmap.new()
  fn classify_kont_reg(name : Var, reg : Reg) {
    match kont_arg_at[reg] {
      None => kont_arg_at[reg] = @hashset.of([name])
      Some(set) => set.insert(name)
    }
  }

  for external in externals {
    guard let @typing.Type::Fun(_, ret) = external.ty.val else {
      _ => @util.die("external non function \{external}")
    }
    match ret {
      Unit | Double => classify_kont_reg(external, A0)
      _ => classify_kont_reg(external, A1)
    }
  }
  for item in kont_arg_at {
    let (kont_reg, externals) = item
    let stub_resolved = cfg.new_named("c_stub_resolved_\{kont_reg}").to_string()
    for external in externals {
      guard let @typing.Type::Fun(args, _) = external.ty.val else {
        _ => @util.die("external non function \{external}")
      }
      let fn_name = external.to_string()
      body.append([Label(fn_name), La(T6, Label("minimbt_" + fn_name))])
      let int_arg_cnt = args
        .iter()
        .filter(fn(arg) { arg != Double && arg != Unit })
        .count()
      let kont_reg = match int_arg_cnt {
        0 => A0
        1 => A1
        2 => A2
        3 => A3
        4 => A4
        5 => A5
        _ => @util.die("too many args for external call")
      }
      // we never return so it's safe to modify the stored regs without backing up
      body.append(
        [
          Comment("store the register holding continuation"),
          Mv(S1, kont_reg),
          J(stub_resolved),
        ],
      )
    }
    // now in swap reg stores our target function's address
    // S1 stores the continuation
    // we just need to first call the target function, then move closure pointer 
    // to a2(where we store continuaion) and then call the continuation
    body.append(
      [
        Label(stub_resolved),
        Comment("Round down the stack for alignment, giving up the space"),
        Li(T5, "-16"),
        And(Sp, Sp, T5),
        Comment("Call the external function"),
        Jalr(T6),
        Comment("Put continuation at \{kont_reg}"),
        Mv(kont_reg, S1),
        Comment("Fetch continuation address"),
        Ld(T1, { base: S1, offset: 0 }),
        Comment("Call continuation"),
        Jr(T1),
      ],
    )
  }
  { name: stub_label, export: false, body }
}
