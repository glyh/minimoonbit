struct JsEmitter {
  clops : @closureps.ClosurePS
  indent : Int
  lid2label : Array[@cps.Var]
  label2lid : @hashmap.T[@cps.Var, Int]
}

pub fn JsEmitter::new(clops : @closureps.ClosurePS) -> JsEmitter {
  // NOTE: idx 0 is reserved for root
  let mut idx = 1
  let lid2label = []
  let label2lid = @hashmap.new()
  for item in clops.fnblocks {
    let label = item.0
    lid2label.push(label)
    label2lid[label] = idx
    idx += 1
  }
  { clops, indent: 0, lid2label, label2lid }
}

pub fn emit_var(var : @cps.Var) -> String {
  "_" + var.id.to_string()
}

pub fn JsEmitter::emit_val(self : JsEmitter, val : @cps.Value) -> String {
  match val {
    Var(var) => emit_var(var)
    Label(var) =>
      match self.label2lid[var] {
        None => "minimbt_" + var.to_string()
        Some(id) => id.to_string()
      }
    Unit => "0"
    Int(i) => i.to_string()
    Double(f) => f.to_string()
  }
}

pub fn JsEmitter::indent(self : JsEmitter) -> JsEmitter {
  { ..self, indent: self.indent + 1 }
}

pub fn JsEmitter::emit_val_array(
  self : JsEmitter,
  vals : Array[@cps.Value]
) -> String {
  let vals_emitted = vals.map(fn(v) { self.emit_val(v) })
  "[" + vals_emitted.join(", ") + "]"
}

pub fn JsEmitter::emit_cps(self : JsEmitter, cps : @cps.Cps) -> String {
  let line_start = "\n" + "  ".repeat(self.indent)
  let mut output = ""
  loop cps {
    Tuple(vals, bind, rest) => {
      let tup_emit = self.emit_val_array(vals)
      output += line_start + "const \{emit_var(bind)} = \{tup_emit};"
      continue rest
    }
    KthTuple(k, tup, bind, rest) => {
      let tup_emit = self.emit_val(tup)
      output += line_start + "const \{emit_var(bind)} = \{tup_emit}[\{k}];"
      continue rest
    }
    Fix(_) => @util.die("there shouldn't be any `Fix`es in ClosurePs")
    If(cond, _then, _else) => {
      let cond_emit = self.emit_val(cond)
      output += line_start + "if (\{cond_emit}) {"
      output += self.indent().emit_cps(_then)
      output += line_start + "} else { "
      output += self.indent().emit_cps(_else)
      output += line_start + "}"
    }
    Prim(Not, [b], bind, rest) => {
      let bool_emit = self.emit_val(b)
      output += line_start + "const \{emit_var(bind)} = 1 - \{bool_emit};"
      continue rest
    }
    Prim(Neg(_), [num], bind, rest) => {
      let num_emit = self.emit_val(num)
      output += line_start + "const \{emit_var(bind)} = -\{num_emit};"
      continue rest
    }
    Prim(Get, [arr, idx], bind, rest) => {
      let arr_emit = self.emit_val(arr)
      let idx_emit = self.emit_val(idx)
      output += line_start +
        "const \{emit_var(bind)} = \{arr_emit}[\{idx_emit}];"
      continue rest
    }
    Prim(Put, [arr, idx, rhs], bind, rest) => {
      let arr_emit = self.emit_val(arr)
      let idx_emit = self.emit_val(idx)
      let rhs_emit = self.emit_val(rhs)
      output += line_start + "const \{emit_var(bind)} = 0;"
      output += line_start + "\{arr_emit}[\{idx_emit}] = \{rhs_emit};"
      continue rest
    }
    Prim(Math(op, ty), [lhs, rhs], bind, rest) => {
      let lhs_emit = self.emit_val(lhs)
      let rhs_emit = self.emit_val(rhs)
      output += match (op, ty) {
        (Add, _) =>
          line_start + "const \{emit_var(bind)} = \{lhs_emit} + \{rhs_emit}"
        (Sub, _) =>
          line_start + "const \{emit_var(bind)} = \{lhs_emit} - \{rhs_emit}"
        (Mul, _) =>
          line_start + "const \{emit_var(bind)} = \{lhs_emit} * \{rhs_emit}"
        (Div, Double) =>
          line_start + "const \{emit_var(bind)} = \{lhs_emit} / \{rhs_emit}"
        (Div, Int) =>
          // NOTE: we may generate -0, but it's fine as when printing we coerce -0 to 0 
          line_start +
          "const \{emit_var(bind)} = Math.trunc(\{lhs_emit} / \{rhs_emit})"
      }
      continue rest
    }
    Prim(Eq, [lhs, rhs], bind, rest) => {
      let lhs_emit = self.emit_val(lhs)
      let rhs_emit = self.emit_val(rhs)
      output += line_start +
        "const \{emit_var(bind)} = \{lhs_emit} === \{rhs_emit} ? 1 : 0;"
      continue rest
    }
    Prim(Le, [lhs, rhs], bind, rest) => {
      let lhs_emit = self.emit_val(lhs)
      let rhs_emit = self.emit_val(rhs)
      output += line_start +
        "const \{emit_var(bind)} = \{lhs_emit} <= \{rhs_emit} ? 1 : 0;"
      continue rest
    }
    Prim(_) as expr => @util.die("malformed prim call \{expr}")
    MakeArray(len, elem, kont_closure) => {
      let len_emit = self.emit_val(len)
      let elem_emit = self.emit_val(elem)
      let kont_emit = self.emit_val(kont_closure)
      output += line_start +
        "return [\{kont_emit}[0], [new Array(\{len_emit}).fill(\{elem_emit}), \{kont_emit}]];"
    }
    Exit => output += line_start + "process.exit(0);"
    App(f, args) => {
      let f_emit = self.emit_val(f)
      let args_emit = self.emit_val_array(args)
      output += line_start + "return [\{f_emit}, \{args_emit}];"
    }
  }
  return output
}

pub fn JsEmitter::emit(self : JsEmitter) -> String {
  let mut output = "compiled_functions = ["
  let line_start = "\n  "

  // 1. emit root
  output += line_start + "function() {"
  output += self.indent().indent().emit_cps(self.clops.root)
  output += line_start + "},"

  // 2. emit all functions
  for label in self.lid2label {
    let cur_fn = self.clops.fnblocks[label].unwrap()
    // i. emit args
    let args_with_closure = cur_fn.args.copy()
    args_with_closure.push(cur_fn.closure)
    let args_with_closure_emitted = args_with_closure.map(emit_var).join(", ")
    output += line_start + "function(\{args_with_closure_emitted}) {"
    // i. emit body
    output += self.indent().indent().emit_cps(cur_fn.body)
    output += line_start + "},"
  }
  output += "\n];\n\n"
  output +=
    #|let current_cont = [0, []]
    #|while(true) {
    #|  const [cont_or_external, args] = current_cont
    #|  if (typeof cont_or_external === "number") { // not external
    #|    current_cont = compiled_functions[cont_or_external](...args)
    #|  } else { // external
    #|    const cont = args.pop()
    #|    const result = cont_or_external(...args)
    #|    current_cont = [cont[0], [result, cont]]
    #|  }
    #|}
  output
}
