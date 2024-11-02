enum WasmTy {
  I32
  F64
  Void // meaning we're not generating any code for such value
} derive(Eq, Hash, Show)

let bit_per_byte = 8

let debug_wasm = false

let params_space_limit : Int = 100 * I32.get_byte_size()

fn WasmTy::from_ty(ty : @typing.Type) -> WasmTy {
  match ty {
    Unit => Void
    Bool | Int | Ptr | Fun(_) | Tuple(_) | Array(_) => I32
    Double => F64
    _ => @util.die("unresolved type \{ty}")
  }
}

fn WasmTy::get_byte_size(self : WasmTy) -> Int {
  match self {
    I32 => 32 / bit_per_byte
    F64 => 64 / bit_per_byte
    Void => 0
  }
}

enum WasmExp {
  Nop
  // I32 exps
  I32Lit(Int)
  I32Add(WasmExp, WasmExp)
  I32Sub(WasmExp, WasmExp)
  I32Mul(WasmExp, WasmExp)
  I32DivS(WasmExp, WasmExp)
  I32TruncSF64(WasmExp)
  I32Eq(WasmExp, WasmExp)
  I32LeS(WasmExp, WasmExp)
  I32Load(WasmExp)
  I32Store(WasmExp, WasmExp)
  // F64 exps
  F64Lit(Double)
  F64Add(WasmExp, WasmExp)
  F64Sub(WasmExp, WasmExp)
  F64Mul(WasmExp, WasmExp)
  F64Div(WasmExp, WasmExp)
  F64ConvertSI32(WasmExp)
  F64Eq(WasmExp, WasmExp)
  F64Le(WasmExp, WasmExp)
  F64Load(WasmExp)
  F64Store(WasmExp, WasmExp)
  F64Abs(WasmExp)
  // Locals
  LocalGet(String, WasmTy)
  LocalDecl(String, WasmTy)
  LocalSet(String, WasmTy, WasmExp)
  GlobalGet(String, WasmTy)
  GlobalSet(String, WasmTy, WasmExp)
  // Control Flows
  If(WasmExp, Array[WasmExp], Array[WasmExp])
  IfStack(Array[WasmExp], Array[WasmExp])
  IfCtrl(WasmExp, Array[WasmExp], Array[WasmExp])
  Call(String, Array[WasmExp])
  CallStack(String)
  //ReturnCont(WasmExp, WasmExp) // This is a psuedo inst
  Loop(String, Array[WasmExp])
  CallIndirect(String, Array[WasmExp], WasmExp)
  Return
  Br(String)
  Block(String, Array[WasmExp])
  BrTable(Array[String], WasmExp)
  Comment(String)
} derive(Show)

let indent_str = "  "

fn WasmExp::emit_str(self : WasmExp, ~indent : Int = 0) -> String {
  fn rec(exp : WasmExp) {
    exp.emit_str(~indent)
  }

  fn rec_indent(exp : WasmExp) {
    exp.emit_str(indent=indent + 1)
  }

  match self {
    Nop => "nop"
    Comment(str) => "(; \{str} ;)"
    I32Lit(i) => "(i32.const \{i})"
    I32Add(lhs, rhs) => "(i32.add \{rec(lhs)} \{rec(rhs)})"
    I32Sub(lhs, rhs) => "(i32.sub \{rec(lhs)} \{rec(rhs)})"
    I32Mul(lhs, rhs) => "(i32.mul \{rec(lhs)} \{rec(rhs)})"
    I32DivS(lhs, rhs) => "(i32.div_s \{rec(lhs)} \{rec(rhs)})"
    I32TruncSF64(num) => "(i32.trunc_f64_s \{rec(num)})"
    I32Eq(lhs, rhs) => "(i32.eq \{rec(lhs)} \{rec(rhs)})"
    I32LeS(lhs, rhs) => "(i32.le_s \{rec(lhs)} \{rec(rhs)})"
    I32Load(addr) => "(i32.load \{rec(addr)})"
    I32Store(addr, value) => "(i32.store \{rec(addr)} \{rec(value)})"
    F64Lit(f) => "(f64.const \{f})"
    F64Add(lhs, rhs) => "(f64.add \{rec(lhs)} \{rec(rhs)})"
    F64Sub(lhs, rhs) => "(f64.sub \{rec(lhs)} \{rec(rhs)})"
    F64Mul(lhs, rhs) => "(f64.mul \{rec(lhs)} \{rec(rhs)})"
    F64Div(lhs, rhs) => "(f64.div \{rec(lhs)} \{rec(rhs)})"
    F64ConvertSI32(num) => "(f64.convert_i32_s \{rec(num)})"
    F64Eq(lhs, rhs) => "(f64.eq \{rec(lhs)} \{rec(rhs)})"
    F64Le(lhs, rhs) => "(f64.le \{rec(lhs)} \{rec(rhs)})"
    F64Load(addr) => "(f64.load \{rec(addr)})"
    F64Store(addr, value) => "(f64.store \{rec(addr)} \{rec(value)})"
    F64Abs(num) => "(f64.abs \{rec(num)})"
    LocalGet(name, _) => "(local.get $\{name})"
    GlobalGet(name, _) => "(global.get $\{name})"
    LocalDecl(name, I32 | Void) => "(local $\{name} i32)" // reserves dummy name for void
    LocalDecl(name, F64) => "(local $\{name} f64)"
    LocalSet(name, _, value) => "(local.set $\{name} \{rec(value)})"
    GlobalSet(name, _, value) => "(global.set $\{name} \{rec(value)})"
    Call(id, args) => {
      let args_str = args.map(rec).join(" ")
      "(call $\{id} \{args_str})"
    }
    CallStack(id) => "(call $\{id})"
    CallIndirect(ty, args, index) => {
      let args_str = args.map(rec).join(" ")
      "(call_indirect (type $\{ty}) \{args_str} \{rec(index)})"
    }
    Return => "return"
    //ReturnCont(fn_id, args) => rec(fn_id) + rec(args)
    Br(target) => "(br $\{target})"
    BrTable(labels, switch_val) => {
      let labels = labels.map(fn(s) { "$" + s }).join(" ")
      "(br_table \{labels} \{rec(switch_val)})"
    }

    // "statements", i.e. we're using new lines here
    If(cond, _then, _else) => {
      let mut output = ""
      output += "(if (result i32) " + rec(cond) + " (then\n"
      output += indent_str.repeat(indent + 1) +
        _then.map(rec_indent).join("\n" + indent_str.repeat(indent + 1))
      output += "\n" + indent_str.repeat(indent) + ") (else\n"
      output += indent_str.repeat(indent + 1) +
        _else.map(rec_indent).join("\n" + indent_str.repeat(indent + 1))
      output += "\n" + indent_str.repeat(indent) + "))"
      output
    }
    IfCtrl(cond, _then, _else) => {
      let mut output = ""
      output += "(if " + rec(cond) + " (then\n"
      output += indent_str.repeat(indent + 1) +
        _then.map(rec_indent).join("\n" + indent_str.repeat(indent + 1))
      output += "\n" + indent_str.repeat(indent) + ") (else\n"
      output += indent_str.repeat(indent + 1) +
        _else.map(rec_indent).join("\n" + indent_str.repeat(indent + 1))
      output += "\n" + indent_str.repeat(indent) + "))"
      output
    }
    IfStack(_then, _else) => {
      let mut output = ""
      output += "(if (result i32) (then\n"
      output += indent_str.repeat(indent + 1) +
        _then.map(rec_indent).join("\n" + indent_str.repeat(indent + 1))
      output += "\n" + indent_str.repeat(indent) + ") (else\n"
      output += indent_str.repeat(indent + 1) +
        _else.map(rec_indent).join("\n" + indent_str.repeat(indent + 1))
      output += "\n" + indent_str.repeat(indent) + "))"
      output
    }
    Block(tag, body) => {
      let mut output = "(block $\{tag}\n"
      output += indent_str.repeat(indent + 1) +
        body.map(rec_indent).join("\n" + indent_str.repeat(indent + 1))
      output += "\n" + indent_str.repeat(indent) + ")"
      output
    }
    Loop(tag, body) => {
      let mut output = "(loop $\{tag}\n"
      output += indent_str.repeat(indent + 1) +
        body.map(rec_indent).join("\n" + indent_str.repeat(indent + 1))
      output += "\n" + indent_str.repeat(indent) + ")"
      output
    }
  }
}

fn WasmExp::collect_locals(
  self : WasmExp,
  target : @hashset.T[(String, WasmTy)]
) -> Unit {
  fn rec(e : WasmExp) {
    e.collect_locals(target)
  }

  match self {
    Nop
    | Comment(_)
    | I32Lit(_) | F64Lit(_) | CallStack(_) | Return | Br(_) | GlobalGet(_) => ()
    I32Add(lhs, rhs)
    | I32Sub(lhs, rhs)
    | I32Mul(lhs, rhs)
    | I32DivS(lhs, rhs)
    | I32Eq(lhs, rhs)
    | I32LeS(lhs, rhs)
    | I32Store(lhs, rhs)
    | F64Add(lhs, rhs)
    | F64Sub(lhs, rhs)
    | F64Mul(lhs, rhs)
    | F64Div(lhs, rhs) | F64Eq(lhs, rhs) | F64Le(lhs, rhs) | F64Store(lhs, rhs) => {
      rec(lhs)
      rec(rhs)
    }
    I32TruncSF64(e)
    | I32Load(e)
    | F64ConvertSI32(e) | F64Load(e) | F64Abs(e) | GlobalSet(_, _, e) => rec(e)
    LocalGet(var, ty) | LocalDecl(var, ty) => target.insert((var, ty))
    LocalSet(var, ty, rhs) => {
      target.insert((var, ty))
      rec(rhs)
    }
    If(cond, _then, _else) | IfCtrl(cond, _then, _else) => {
      rec(cond)
      _then.each(rec)
      _else.each(rec)
    }
    IfStack(_then, _else) => {
      _then.each(rec)
      _else.each(rec)
    }
    Call(_, args) => args.each(rec)
    CallIndirect(_, args, index) => {
      rec(index)
      args.each(rec)
    }
    //ReturnCont(id, args) => {
    //  rec(id)
    //  rec(args)
    //}
    Loop(_, body) | Block(_, body) => body.each(rec)
    BrTable(_, switched) => rec(switched)
  }
}

struct WasmFunc {
  name : String
  func_id : Int?
  params : Array[(WasmTy, String)]
  body : Array[WasmExp]
  is_entrance : Bool
  closure_name : String?
}

fn WasmFunc::emit_str(self : WasmFunc, target : WasmTarget) -> String {
  let mut output = ""
  if self.is_entrance {
    output += "(func $\{self.name}"
  } else {
    output += "(func $\{self.name} (;ID: \{self.func_id};) (type $wrapfn) (;\{self.params};) (result i32)"
  }
  let fn_prologue = []
  let mut idx = 0
  let mut offset = 0
  for param in self.params {
    match param.0 {
      Void => fn_prologue.push(LocalSet(param.1, I32, I32Lit(0)))
      I32 =>
        fn_prologue.push(
          LocalSet(
            param.1,
            I32,
            I32Load(I32Add(GlobalGet("params", I32), I32Lit(offset))),
          ),
        )
      F64 =>
        fn_prologue.push(
          LocalSet(
            param.1,
            F64,
            F64Load(I32Add(GlobalGet("params", I32), I32Lit(offset))),
          ),
        )
    }
    offset += param.0.get_byte_size()
    idx += 1
  }
  match self.closure_name {
    Some(name) =>
      fn_prologue.append(
        [
          Comment("Load closure pointer"),
          LocalSet(
            name,
            I32,
            I32Load(I32Add(GlobalGet("params", I32), I32Lit(offset))),
          ),
        ],
      )
    _ => ()
  }
  let locals = @hashset.new()
  fn _collect_label(exp : WasmExp) {
    exp.collect_locals(locals)
  }

  fn_prologue.each(_collect_label)
  self.body.each(_collect_label)
  //if self.name == "main" {
  //  println("\{locals}")
  //  println("\{self.body}")
  //}
  let local_decls = locals
    .iter()
    .map(fn { (name, ty) => LocalDecl(name, ty) })
    .collect()
  let const_local_load = locals
    .iter()
    .filter(fn { (name, _) => target.closure_id.contains(name) })
    .flat_map(
      fn {
        (name, ty) =>
          [
            Comment("load const closure \{name}"),
            LocalSet(
              name,
              ty,
              I32Add(
                GlobalGet("const_clo_arr", I32),
                I32Lit(target.closure_id[name].unwrap() * I32.get_byte_size()),
              ),
            ),
          ].iter()
      },
    )
    .collect()
  let all_insts = local_decls + const_local_load + fn_prologue + self.body
  for exp in all_insts {
    output += "\n" + indent_str + exp.emit_str(indent=1)
  }
  output += "\n)"
  output
}

struct WasmTarget {
  funcs : Array[WasmFunc]
  fn_table : Array[String]
  untabled_externals : @hashset.T[String]
  externals : Array[String]
  closure_id : @hashmap.T[String, Int]
  function_id : @hashmap.T[String, Int]
}

pub fn WasmTarget::emit_str(self : WasmTarget) -> String {
  let mut output = "(module"
  let importeds = self.externals.copy() +
    self.untabled_externals.iter().collect()
  fn ensure_imported(name) {
    if not(importeds.contains(name)) {
      importeds.push(name)
    }
  }

  if debug_wasm {
    ensure_imported("print_int")
    ensure_imported("print_char")
    ensure_imported("print_endline")
  }
  for imported in importeds {
    let external_name = "minimbt_" + imported
    guard let Some(Fun(args, ret)) = @typing.externals[imported] else {
      None => @util.die("no such external \{external_name}")
      Some(_) => @util.die("non function external \{external_name}")
    }
    let mut fn_ty = ""
    for arg in args {
      fn_ty += "(param"
      match WasmTy::from_ty(arg) {
        I32 | Void => fn_ty += " i32"
        F64 => fn_ty += " f64"
      }
      fn_ty += ") "
    }
    match WasmTy::from_ty(ret) {
      I32 => fn_ty += "(result i32) "
      F64 => fn_ty += "(result f64) "
      Void => ()
    }
    output += "\n(import \"moonbit\" \"\{external_name}\" (func $\{external_name} \{fn_ty}))"
  }
  output +=
    #|
    #|(import "moonbit" "memory" (memory 10))
    #|(type $wrapfn (func (result i32)))
    #|(global $const_clo_arr (mut i32) (i32.const 0))
    #|(global $params (mut i32) (i32.const 0))
  output += "\n" +
    self.funcs.map(fn(f) { WasmFunc::emit_str(f, self) }).join("\n")
  // NOTE: we reserve one more slot so offset matches
  output += "\n(table $funcs \{self.fn_table.length() + 1} funcref)"
  output += "\n(elem (i32.const 1) " +
    self.fn_table.map(fn(id) { "$" + id }).join(" ") +
    ")"
  let func_wrapper : WasmFunc = {
    name: "main",
    params: [],
    body: [],
    is_entrance: true,
    closure_name: None,
    func_id: None,
  }
  let mut external_detect = [
    BrTable(self.externals, I32Sub(I32Lit(-1), LocalGet("kont_id", I32))),
  ]
  external_detect = self.externals.fold(
    init=external_detect,
    fn(acc, external_name) {
      guard let Some(Fun(args, ret)) = @typing.externals[external_name] else {
        None => @util.die("no such external \{external_name}")
        Some(_) => @util.die("non function external \{external_name}")
      }
      let mut cur_offset = 0
      let prepare_args = args.map(
        fn(ty) {
          let addr = I32Add(GlobalGet("params", I32), I32Lit(cur_offset))
          let arg_ty = WasmTy::from_ty(ty)
          cur_offset += arg_ty.get_byte_size()
          match arg_ty {
            I32 => I32Load(addr)
            F64 => F64Load(addr)
            Void => I32Lit(0)
          }
        },
      )
      let ret_ty = WasmTy::from_ty(ret)
      let call_and_finalize = [
        match ret_ty {
          I32 => LocalSet("i_ret", I32, CallStack("minimbt_" + external_name))
          F64 => LocalSet("f_ret", F64, CallStack("minimbt_" + external_name))
          Void => CallStack("minimbt_" + external_name)
        },
        LocalSet(
          "kont_closure",
          I32,
          I32Load(I32Add(LocalGet("params", I32), I32Lit(cur_offset))),
        ),
        match ret_ty {
          I32 => I32Store(GlobalGet("params", I32), LocalGet("i_ret", I32))
          F64 => F64Store(GlobalGet("params", I32), LocalGet("f_ret", F64))
          Void => Nop
        },
        I32Store(
          I32Add(GlobalGet("params", I32), I32Lit(ret_ty.get_byte_size())),
          LocalGet("kont_closure", I32),
        ),
        LocalSet("kont_id", I32, I32Load(LocalGet("kont_closure", I32))),
        Br("loop_label"),
      ]
      [Block(external_name, acc)] + prepare_args + call_and_finalize
    },
  )
  let const_closures = self.closure_id.iter().collect()
  const_closures.sort_by_key(fn { (_, id) => id })
  func_wrapper.body.append(
    [
      GlobalSet(
        "params",
        I32,
        Call("minimbt_malloc", [I32Lit(params_space_limit)]),
      ),
      GlobalSet(
        "const_clo_arr",
        I32,
        Call(
          "minimbt_malloc",
          [I32Lit(self.closure_id.size() * I32.get_byte_size())],
        ),
      ),
    ],
  )
  let mut offset = 0
  for const_closure in const_closures {
    let (clo_name, _) = const_closure
    func_wrapper.body.append(
      [
        Comment("allocate space for \{clo_name}"),
        I32Store(
          I32Add(GlobalGet("const_clo_arr", I32), I32Lit(offset)),
          I32Lit(self.function_id[clo_name].unwrap()),
        ),
      ],
    )
    offset += I32.get_byte_size()
  }
  let trace_cont = if debug_wasm {
    [
      Comment("DEBUG HELPER STARTS"),
      Call("minimbt_print_char", [I32Lit(91)]),
      Call("minimbt_print_int", [LocalGet("kont_id", I32)]),
      Call("minimbt_print_char", [I32Lit(58)]),
      Call("minimbt_print_char", [I32Lit(32)]),
      Call(
        "minimbt_print_int",
        [I32Load(I32Add(GlobalGet("params", I32), I32Lit(0)))],
      ),
      Call("minimbt_print_char", [I32Lit(44)]),
      Call(
        "minimbt_print_int",
        [I32Load(I32Add(GlobalGet("params", I32), I32Lit(4)))],
      ),
      Call("minimbt_print_char", [I32Lit(44)]),
      Call(
        "minimbt_print_int",
        [I32Load(I32Add(GlobalGet("params", I32), I32Lit(8)))],
      ),
      Call("minimbt_print_char", [I32Lit(93)]),
      Call("minimbt_print_endline", []),
      Comment("DEBUG HELPER ENDS"),
    ]
  } else {
    []
  }
  func_wrapper.body.append(
    [
      LocalSet("kont_id", I32, Call("minimbt_main", [])),
      Loop(
        "loop_label",
        trace_cont +
        [
          IfCtrl(
            I32Eq(LocalGet("kont_id", I32), I32Lit(0)),
            [Return],
            [
              IfCtrl(
                I32LeS(LocalGet("kont_id", I32), I32Lit(-1)),
                external_detect,
                [
                  LocalSet(
                    "kont_id",
                    I32,
                    CallIndirect("wrapfn", [], LocalGet("kont_id", I32)),
                  ),
                  Br("loop_label"),
                ],
              ),
            ],
          ),
        ],
      ),
    ],
  )
  output += "\n" + func_wrapper.emit_str(self)
  output +=
    #|
    #|(start $main)
  output += "\n)"
  output
}
