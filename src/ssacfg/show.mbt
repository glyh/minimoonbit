impl Show for SsaCfg with output(self, logger) {
  logger.write_string(self.to_string())
}

pub fn SsaCfg::to_string(self : SsaCfg) -> String {
  let mut output = ""
  for item in self.blocks {
    let (name, def) = item
    output += "\{name}"
    match self.fn_args[name] {
      None => output += "\n"
      Some(args) => output += " \{args}\n"
    }
    for inst in def.insts {
      output += "  \{inst}\n"
    }
    output += "  \{def.last_inst.val}\n"
  }
  output
}
