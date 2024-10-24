struct JsEmitter {
  clops : @closureps.ClosurePS
  indent : Int
}

pub fn JsEmitter::new(clops : @closureps.ClosurePS) -> JsEmitter {
  { clops, indent: 0 }
}

pub fn emit_var(var : @cps.Var) -> String {
  if var.id < 0 {
    "minimbt_" + var.name.val.unwrap()
  } else {
    var.to_string()
  }
}

pub fn emit_val(val : @cps.Value) -> String {
  match val {
    //Var(var) => emit_var(var)
    Var(var) | Label(var) => emit_var(var)
    //match self.label2lid[var] {
    //  None => 
    //  Some(id) => id.to_string()
    //}
    Unit => "0"
    Int(i) => i.to_string()
    Bool(b) => b.to_string()
    Double(f) => f.to_string()
  }
}

pub fn JsEmitter::indent(self : JsEmitter) -> JsEmitter {
  { ..self, indent: self.indent + 1 }
}

pub fn JsEmitter::emit_val_array(vals : Array[@cps.Value]) -> String {
  let vals_emitted = vals.map(emit_val)
  "[" + vals_emitted.join(", ") + "]"
}

pub fn JsEmitter::emit_cps(self : JsEmitter, cps : @cps.Cps) -> String {
  let line_start = "\n" + "  ".repeat(self.indent)
  let mut output = ""
  loop cps {
    Tuple(vals, bind, rest) => {
      let tup_emit = emit_val_array(vals)
      output += line_start + "const \{emit_var(bind)} = \{tup_emit};"
      continue rest
    }
    KthTuple(k, tup, bind, rest) => {
      let tup_emit = emit_val(tup)
      output += line_start + "const \{emit_var(bind)} = \{tup_emit}[\{k}];"
      continue rest
    }
    Fix(_) => @util.die("there shouldn't be any `Fix`es in ClosurePs")
    If(cond, _then, _else) => {
      let cond_emit = emit_val(cond)
      output += line_start + "if (\{cond_emit}) {"
      output += self.indent().emit_cps(_then)
      output += line_start + "} else { "
      output += self.indent().emit_cps(_else)
      output += line_start + "}"
    }
    IfEq(lhs, rhs, _then, _else) => {
      let lhs_emit = emit_val(lhs)
      let rhs_emit = emit_val(rhs)
      output += line_start + "if (\{lhs_emit} === \{rhs_emit}) {"
      output += self.indent().emit_cps(_then)
      output += line_start + "} else { "
      output += self.indent().emit_cps(_else)
      output += line_start + "}"
    }
    IfLe(lhs, rhs, _then, _else) => {
      let lhs_emit = emit_val(lhs)
      let rhs_emit = emit_val(rhs)
      output += line_start + "if (\{lhs_emit} <= \{rhs_emit}) {"
      output += self.indent().emit_cps(_then)
      output += line_start + "} else { "
      output += self.indent().emit_cps(_else)
      output += line_start + "}"
    }
    Prim(Not, [b], bind, rest) => {
      let bool_emit = emit_val(b)
      output += line_start + "const \{emit_var(bind)} = !\{bool_emit};"
      continue rest
    }
    Prim(Neg(_), [num], bind, rest) => {
      let num_emit = emit_val(num)
      output += line_start + "const \{emit_var(bind)} = -\{num_emit};"
      continue rest
    }
    Prim(Get, [arr, idx], bind, rest) => {
      let arr_emit = emit_val(arr)
      let idx_emit = emit_val(idx)
      output += line_start +
        "const \{emit_var(bind)} = \{arr_emit}[\{idx_emit}];"
      continue rest
    }
    Prim(Put, [arr, idx, rhs], bind, rest) => {
      let arr_emit = emit_val(arr)
      let idx_emit = emit_val(idx)
      let rhs_emit = emit_val(rhs)
      output += line_start + "const \{emit_var(bind)} = 0;"
      output += line_start + "\{arr_emit}[\{idx_emit}] = \{rhs_emit};"
      continue rest
    }
    Prim(Math(op, ty), [lhs, rhs], bind, rest) => {
      let lhs_emit = emit_val(lhs)
      let rhs_emit = emit_val(rhs)
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
      let lhs_emit = emit_val(lhs)
      let rhs_emit = emit_val(rhs)
      output += line_start +
        "const \{emit_var(bind)} = \{lhs_emit} === \{rhs_emit};"
      continue rest
    }
    Prim(Le, [lhs, rhs], bind, rest) => {
      let lhs_emit = emit_val(lhs)
      let rhs_emit = emit_val(rhs)
      output += line_start +
        "const \{emit_var(bind)} = \{lhs_emit} <= \{rhs_emit};"
      continue rest
    }
    Prim(IntOfFloat, [f], bind, rest) => {
      let f_emit = emit_val(f)
      output += line_start + "const \{emit_var(bind)} = Math.trunc(\{f_emit});"
      continue rest
    }
    Prim(FloatOfInt, [i], bind, rest) => {
      let i_emit = emit_val(i)
      output += line_start + "const \{emit_var(bind)} = \{i_emit};"
      continue rest
    }
    Prim(AbsFloat, [f], bind, rest) => {
      let f_emit = emit_val(f)
      output += line_start + "const \{emit_var(bind)} = Math.abs(\{f_emit});"
      continue rest
    }
    Prim(_) as expr => @util.die("malformed prim call \{expr}")
    MakeArray(len, elem, kont_closure) => {
      let len_emit = emit_val(len)
      let elem_emit = emit_val(elem)
      let kont_emit = emit_val(kont_closure)
      output += line_start +
        "return [\{kont_emit}[0], [new Array(\{len_emit}).fill(\{elem_emit}), \{kont_emit}]];"
    }
    Exit => output += line_start + "process.exit(0);"
    App(f, args) => {
      let f_emit = emit_val(f)
      let args_emit = emit_val_array(args)
      output += line_start + "return [\{f_emit}, \{args_emit}];"
    }
  }
  return output
}

pub fn JsEmitter::emit(self : JsEmitter) -> String {
  let line_start = "\n"
  let mut output = ""

  // A. emit all const closures
  output +=
    #|// All const closures
  for f in self.clops.func_no_free_vars {
    let clo = self.clops.label_to_closure[f].unwrap()
    output += line_start + "const \{clo} = [\{emit_val(Label(f))}, []]"
  }
  output += line_start
  // B. emit all functions
  // 1. emit root
  output += line_start + "function root() { // root"
  output += self.indent().indent().emit_cps(self.clops.root)
  output += line_start + "}"
  // 2. emit all functions
  for item in self.clops.fnblocks {
    let (label, cur_fn) = item
    // i. emit args
    let args_with_closure = cur_fn.args.copy()
    args_with_closure.push(cur_fn.closure)
    let args_with_closure_emitted = args_with_closure.join(", ")
    output += line_start + "function \{label}(\{args_with_closure_emitted}) {"
    // ii. emit body
    output += self.indent().indent().emit_cps(cur_fn.body)
    output += line_start + "}"
  }
  output += "\n\n"
  output +=
    #|export default function () {
    #|  let current_cont = root();
    #|  while(true) {
    #|    const [cont_or_external, args] = current_cont;
    #|    if (cont_or_external.name.startsWith('_minimbt')) { // not external
    #|      current_cont = cont_or_external(...args);
    #|    } else { // external
    #|      const cont = args.pop();
    #|      const result = cont_or_external(...args);
    #|      current_cont = [cont[0], [result, cont]];
    #|    }
    #|  }
    #|}
  output
}
