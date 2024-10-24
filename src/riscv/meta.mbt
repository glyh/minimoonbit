fn generate_meta(cfg : @ssacfg.SsaCfg) -> AssemblyFunction {
  let body : Array[RvAsm] = []

  // Generate all constant closures
  body.push(Section(".data"))
  for fn_label in cfg.func_no_free_vars {
    body.append(
      [
        Label(cfg.label_to_closure[fn_label].unwrap().to_string()),
        DWord(generate_name(fn_label)),
      ],
    )
  }
  body.push(Section(".bss"))
  body.append(
    [
      Comment("Align at 8-byte for x64"),
      Align(3),
      Label("stack_space"),
      Comment("Skips 1 GiB For stack"),
      Skip(1024 * 1024 * 1024),
      Label("stack_space_end"),
    ],
  )
  body.push(Section(".text"))
  { name: "meta", export: false, body }
}
