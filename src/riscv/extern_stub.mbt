fn generate_stub(cfg : @ssacfg.SsaCfg) -> AssemblyFunction {
  // generate stub for CPS / C-Calling convention interop
  let stub_label = cfg.new_named("c_stub").to_string()
  let stub_resolved_i = cfg.new_named("c_stub_resolved_i").to_string()
  let body : Array[RvAsm] = []
  let externals = collect_externals(cfg)
  // generate stubs for function returning ints
  for external in externals.returns_i {
    guard let @typing.Type::Fun(args, _) = external.ty.val else {
      _ => @util.die("external non function \{external}")
    }
    let fn_name = external.to_string()
    body.append([Label(fn_name), La(T6, Label("minimbt_" + fn_name))])
    let int_arg_cnt = args.iter().filter(fn(arg) { arg != Double }).count()
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
        J(stub_resolved_i),
      ],
    )
  }
  // now in swap reg stores our target function's address
  // S1 stores the continuation
  // we just need to first call the target function, then move closure pointer 
  // to a2(where we store continuaion) and then call the continuation
  body.append(
    [
      Label(stub_resolved_i),
      Comment("call the external function"),
      Jalr(T6),
      Comment("A0 holds result, put continuation as 2nd arg"),
      Mv(A1, S1),
      Comment("fetch continuation address"),
      Ld(T1, { base: S1, offset: 0 }),
      Comment("call continuation"),
      Jr(T1),
    ],
  )
  // generates stub for function returning floats
  let stub_resolved_f = cfg.new_named("c_stub_resolved_f").to_string()
  for external in externals.returns_f {
    guard let @typing.Type::Fun(args, _) = external.ty.val else {
      _ => @util.die("external non function \{external}")
    }
    let fn_name = external.to_string()
    body.append([Label(fn_name), La(T6, Label("minimbt_" + fn_name))])
    let int_arg_cnt = args.iter().filter(fn(arg) { arg != Double }).count()
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
        J(stub_resolved_f),
      ],
    )
  }
  // now in swap reg stores our target function's address
  // S1 stores the continuation
  // we just need to first call the target function, then move closure pointer 
  // to a2(where we store continuaion) and then call the continuation
  body.append(
    [
      Label(stub_resolved_f),
      Comment("call the external function"),
      Jalr(T6),
      Comment("Fa0 holds result, put continuation as 2nd arg"),
      Mv(A0, S1),
      Comment("fetch continuation address"),
      Ld(T1, { base: S1, offset: 0 }),
      Comment("call continuation"),
      Jr(T1),
    ],
  )
  { name: stub_label, export: false, body }
}
