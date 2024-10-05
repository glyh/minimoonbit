impl Show for Var with output(self, logger) {
  logger.write_string(self.to_string())
}

pub fn Var::to_string(self : Var) -> String {
  match self.name.val {
    None => "?\{self.id}"
    Some(n) => if self.id < 0 { n } else { "\{n}.\{self.id}" }
  }
}

impl Show for Value with output(self, logger) {
  logger.write_string(self.to_string())
}

pub fn Value::to_string(self : Value) -> String {
  match self {
    Var(v) => v.to_string()
    Label(v) => ":" + v.to_string()
    Unit => "()"
    Int(i) => i.to_string()
    Double(f) => f.to_string()
  }
}

impl Show for Cps with output(self, logger) {
  logger.write_string(self.to_string())
}

pub fn Cps::to_string(self : Cps) -> String {
  to_str(self)
}

fn to_str(cps : Cps, ~ident : String = "") -> String {
  fn rec(c : Cps) {
    to_str(c, ~ident)
  }

  match cps {
    Record(arr, bind, rest) => ident + "\{bind} = \{arr}\n" + rec(rest)
    Select(idx, v, bind, rest) => ident + "\{bind} = \{v}[\{idx}]\n" + rec(rest)
    Offset(idx, v, bind, rest) => ident + "\{bind} = \{v}+\{idx}\n" + rec(rest)
    Fix(name, args, body, rest) =>
      ident +
      "fn \{name}(\{args}) {\n" +
      to_str(body, ident=ident + "  ") +
      "\n" +
      ident +
      "}\n" +
      rec(rest)
    Switch(v, branches) =>
      ident +
      "switch(\{v}){\n" +
      branches.map(fn { c => to_str(c, ident=ident + "  ") }).join(";\n") +
      "\n" +
      ident +
      "}"
    Prim(op, args, bind, rest) =>
      ident + "prim \{bind} = \{op}(\{args})\n" + rec(rest)
    App(f, args) => ident + "\{f}(\{args})"
    Just(v) => ident + "return \{v}"
  }
}
