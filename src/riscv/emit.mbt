pub struct AssemblyFunction {
  name : String
  export : Bool
  body : Array[RvAsm]
}

pub fn AssemblyFunction::output(
  self : AssemblyFunction,
  logger : Logger
) -> Unit {
  let name = self.name
  if self.export {
    logger.write_string(".global \{name}\n")
  }
  for asm in self.body {
    Show::output(asm, logger)
    logger.write_string("\n")
  }
}

pub fn emit(cfg : @ssacfg.SsaCfg) -> Array[AssemblyFunction] {
  let cfg = before_alloc(cfg)
  let output = []
  output.push(generate_meta(cfg))
  output.push(generate_stub(cfg))
  // genereating asm code for all functions
  for item in cfg.fn_args {
    let (fn_label, _) = item
    let allocation = reg_allocate_on_fn(cfg, fn_label)
    let codegen_blk = CodegenBlock::new(cfg, allocation, fn_label)
    codegen_blk.codegen()
    output.push(codegen_blk.cur_fn.val)
  }
  output
}
