impl Show for ClosurePS with output(self, logger) {
  logger.write_string(self.to_string())
}

pub fn ClosurePS::to_string(self : ClosurePS) -> String {
  let mut output = ""
  for item in self.fnblocks.iter() {
    let (name, def) = item
    output += "\{name}, args: \{def.args}, freevars: \{def.free_vars}, cloptr: \{def.closure}\n"
    output += "\{def.body}\n\n"
  }
  output += "[root]\n\{self.root}\n"
  output
}
