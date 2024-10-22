pub impl Show for Var with output(self, logger) {
  logger.write_string(self.to_string())
}

pub fn Var::to_string(self : Var) -> String {
  match self.name.val {
    None => "_minimbt_\{self.id}"
    Some(n) => if self.id < 0 { n } else { "_minimbt_\{n}_\{self.id}" }
  }
}

pub impl Show for Value with output(self, logger) {
  logger.write_string(self.to_string())
}

pub fn Value::to_string(self : Value) -> String {
  match self {
    Bool(b) => b.to_string()
    Var(v) => v.to_string()
    Label(v) => ":" + v.to_string()
    Unit => "()"
    Int(i) => i.to_string()
    Double(f) => f.to_string()
  }
}

pub impl Show for Cps with output(self, logger) {
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
    Tuple(arr, bind, rest) => ident + "\{bind} = \{arr}\n" + rec(rest)
    KthTuple(idx, v, bind, rest) =>
      ident + "\{bind} = \{v}.\{idx}\n" + rec(rest)
    Fix(name, args, body, rest) =>
      ident +
      "fn \{name}(\{args}) {\n" +
      to_str(body, ident=ident + "  ") +
      "\n" +
      ident +
      "}\n" +
      rec(rest)
    IfEq(lhs, rhs, _then, _else) =>
      ident +
      "if(\{lhs} == \{rhs}){\n" +
      to_str(_then, ident=ident + "  ") +
      "\n" +
      ident +
      "} else {\n" +
      to_str(_else, ident=ident + "  ") +
      "\n" +
      ident +
      "}"
    IfLe(lhs, rhs, _then, _else) =>
      ident +
      "if(\{lhs} <= \{rhs}){\n" +
      to_str(_then, ident=ident + "  ") +
      "\n" +
      ident +
      "} else {\n" +
      to_str(_else, ident=ident + "  ") +
      "\n" +
      ident +
      "}"
    If(cond, _then, _else) =>
      ident +
      "if(\{cond}){\n" +
      to_str(_then, ident=ident + "  ") +
      "\n" +
      ident +
      "} else {\n" +
      to_str(_else, ident=ident + "  ") +
      "\n" +
      ident +
      "}"
    Prim(op, args, bind, rest) =>
      ident + "prim \{bind} = \{op}(\{args})\n" + rec(rest)
    App(f, args) => ident + "\{f}(\{args})"
    MakeArray(len, elem, kont) =>
      ident + "Array::make(\{len}, \{elem}).then(\{kont})"
    Exit => ident + "exit"
  }
}
