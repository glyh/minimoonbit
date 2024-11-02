struct WasmEmitter {
  clops : @closureps.ClosurePS
  function_id : @hashmap.T[@cps.Var, Int]
  used_externals : Array[String]
  untabled_externals : @hashset.T[String]
}

pub fn WasmEmitter::new_named(
  self : WasmEmitter,
  name : String,
  ~ty : @typing.Type = Unit
) -> @cps.Var {
  self.clops.counter.val += 1
  { name: { val: Some(name) }, id: self.clops.counter.val, ty: { val: ty } }
}

pub fn WasmEmitter::new(clops : @closureps.ClosurePS) -> WasmEmitter {
  let function_id = @hashmap.new()
  // NOTE: 
  // 1. negative function ids refer to external
  // 2. positive function ids refer to generated functions
  // 3. 0 refers to exit
  clops.fnblocks.eachi(fn(i, k, _v) { function_id[k] = i + 1 })
  {
    clops,
    function_id,
    used_externals: [],
    untabled_externals: @hashset.of(["malloc"]),
  }
}

pub fn WasmEmitter::emit_var(self : WasmEmitter, var : @cps.Var) -> WasmExp {
  if var.id < 0 {
    let ext_name = var.name.val.unwrap()
    match self.used_externals.search(ext_name) {
      None => {
        let id = self.used_externals.length()
        self.used_externals.push(ext_name)
        I32Lit(-1 - id)
      }
      Some(id) => I32Lit(-1 - id)
    }
  } else {
    match self.function_id[var] {
      None => LocalGet(var.to_string(), WasmTy::from_ty(var.ty.val))
      Some(id) => I32Lit(id)
    }
  }
}

pub fn WasmEmitter::emit_val(self : WasmEmitter, val : @cps.Value) -> WasmExp {
  match val {
    Var(var) | Label(var) => self.emit_var(var)
    Unit => I32Lit(0)
    Int(i) => I32Lit(i)
    Bool(true) => I32Lit(1)
    Bool(false) => I32Lit(0)
    Double(f) => F64Lit(f)
  }
}

fn get_tuple_size_ty(tup_ty : Array[@typing.Type]) -> Int {
  let mut result = 0
  loop tup_ty[:] {
    [] => break result
    [ty, .. as tup_ty] => {
      result += WasmTy::from_ty(ty).get_byte_size()
      continue tup_ty
    }
  }
}

fn get_tuple_size(tup : Array[@cps.Value]) -> Int {
  get_tuple_size_ty(tup.map(@cps.Value::get_type))
}

fn get_tuple_offset(tup : Array[@typing.Type], k : Int) -> Int {
  let mut result = 0
  let mut cur_idx = 0
  loop tup[:] {
    [] => @util.die("no value available at \{k}th position in \{tup}")
    [t, .. as tup] =>
      if cur_idx == k {
        break result
      } else {
        result += WasmTy::from_ty(t).get_byte_size()
        cur_idx += 1
        continue tup
      }
  }
}

fn WasmEmitter::emit_store(
  self : WasmEmitter,
  addr : WasmExp,
  val : @cps.Value,
  output : Array[WasmExp]
) -> Unit {
  let exp_to_store = self.emit_val(val)
  match WasmTy::from_ty(val.get_type()) {
    Void => ()
    I32 => output.push(I32Store(addr, exp_to_store))
    F64 => output.push(F64Store(addr, exp_to_store))
  }
}

fn emit_load(addr : WasmExp, bind : @cps.Var, output : Array[WasmExp]) -> Unit {
  let bind_sym = bind.to_string()
  match WasmTy::from_ty(bind.ty.val) {
    Void => output.push(LocalSet(bind_sym, Void, I32Lit(0)))
    I32 => output.push(LocalSet(bind_sym, I32, I32Load(addr)))
    F64 => output.push(LocalSet(bind_sym, F64, F64Load(addr)))
  }
}

fn WasmEmitter::emit_tuple(
  self : WasmEmitter,
  vals : Array[@cps.Value],
  bind : @cps.Var,
  output : Array[WasmExp]
) -> Unit {
  let tuple_size = get_tuple_size(vals)
  let bind_sym = bind.to_string()
  output.push(
    LocalSet(bind_sym, I32, Call("minimbt_malloc", [I32Lit(tuple_size)])),
  )
  let mut offset = 0
  vals.each(
    fn(val) {
      self.emit_store(
        I32Add(LocalGet(bind_sym, I32), I32Lit(offset)),
        val,
        output,
      )
      offset += WasmTy::from_ty(val.get_type()).get_byte_size()
    },
  )
}

fn WasmEmitter::emit_kth_tuple(
  self : WasmEmitter,
  k : Int,
  tup : @cps.Value,
  bind : @cps.Var,
  output : Array[WasmExp]
) -> Unit {
  let tup_emitted = self.emit_val(tup)
  let offset = match tup.get_type() {
    Tuple(tys) => get_tuple_offset(tys, k)
    Fun(_, _) => // closure
      match k {
        0 => 0
        1 => I32.get_byte_size()
        k => @util.die("attempt to index closure at index \{k}")
      }
    ty => @util.die("expect tuple for `kth tuple` got \{ty}")
  }
  emit_load(I32Add(tup_emitted, I32Lit(offset)), bind, output)
}

fn WasmEmitter::emit_call(
  self : WasmEmitter,
  called : @cps.Value,
  args : Array[@cps.Value],
  output : Array[WasmExp]
) -> Unit {
  //let args_tup = self.new_named(
  //  "args_tup",
  //  ty=Tuple(args.map(@cps.Value::get_type)),
  //)
  //self.emit_tuple(args, args_tup, output)
  let mut offset = 0
  for arg in args {
    let arg_ty = WasmTy::from_ty(arg.get_type())
    match arg_ty {
      I32 =>
        output.push(
          I32Store(
            I32Add(GlobalGet("params", I32), I32Lit(offset)),
            self.emit_val(arg),
          ),
        )
      F64 =>
        output.push(
          F64Store(
            I32Add(GlobalGet("params", I32), I32Lit(offset)),
            self.emit_val(arg),
          ),
        )
      Void => ()
    }
    offset += arg_ty.get_byte_size()
  }
  output.push(self.emit_val(called))
}

pub fn WasmEmitter::emit_cps(
  self : WasmEmitter,
  cps : @cps.Cps,
  output : Array[WasmExp]
) -> Unit {
  loop cps {
    Tuple(vals, bind, rest) => {
      self.emit_tuple(vals, bind, output)
      continue rest
    }
    KthTuple(k, tup, bind, rest) => {
      self.emit_kth_tuple(k, tup, bind, output)
      continue rest
    }
    Fix(_) => @util.die("there shouldn't be any `Fix`es in ClosurePs")
    If(cond, _then, _else) => {
      let _then_emitted = []
      self.emit_cps(_then, _then_emitted)
      let _else_emitted = []
      self.emit_cps(_else, _else_emitted)
      output.push(If(self.emit_val(cond), _then_emitted, _else_emitted))
      break
    }
    IfEq(lhs, rhs, _then, _else) => {
      match lhs.get_type() {
        Double => output.push(F64Eq(self.emit_val(lhs), self.emit_val(rhs)))
        Unit => output.push(I32Lit(1))
        _ => output.push(I32Eq(self.emit_val(lhs), self.emit_val(rhs)))
      }
      let _then_emitted = []
      self.emit_cps(_then, _then_emitted)
      let _else_emitted = []
      self.emit_cps(_else, _else_emitted)
      output.push(IfStack(_then_emitted, _else_emitted))
      break
    }
    IfLe(lhs, rhs, _then, _else) => {
      match lhs.get_type() {
        Double => output.push(F64Le(self.emit_val(lhs), self.emit_val(rhs)))
        Unit => output.push(I32Lit(1))
        _ => output.push(I32LeS(self.emit_val(lhs), self.emit_val(rhs)))
      }
      let _then_emitted = []
      self.emit_cps(_then, _then_emitted)
      let _else_emitted = []
      self.emit_cps(_else, _else_emitted)
      output.push(IfStack(_then_emitted, _else_emitted))
      break
    }
    Prim(Not, [b], bind, rest) => {
      let bind_sym = bind.to_string()
      output.push(LocalSet(bind_sym, I32, I32Sub(I32Lit(1), self.emit_val(b))))
      continue rest
    }
    Prim(Neg(Int), [num], bind, rest) => {
      let bind_sym = bind.to_string()
      output.push(
        LocalSet(bind_sym, I32, I32Sub(I32Lit(0), self.emit_val(num))),
      )
      continue rest
    }
    Prim(Neg(Double), [num], bind, rest) => {
      let bind_sym = bind.to_string()
      output.push(
        LocalSet(bind_sym, F64, F64Sub(F64Lit(0), self.emit_val(num))),
      )
      continue rest
    }
    Prim(Get, [arr, idx], bind, rest) => {
      let elem_size = WasmTy::from_ty(bind.ty.val).get_byte_size()
      let offset = I32Mul(I32Lit(elem_size), self.emit_val(idx))
      emit_load(I32Add(self.emit_val(arr), offset), bind, output)
      continue rest
    }
    Prim(Put, [arr, idx, rhs], bind, rest) => {
      let elem_size = WasmTy::from_ty(rhs.get_type()).get_byte_size()
      let offset = I32Mul(I32Lit(elem_size), self.emit_val(idx))
      self.emit_store(I32Add(self.emit_val(arr), offset), rhs, output)
      output.push(LocalSet(bind.to_string(), I32, I32Lit(0)))
      continue rest
    }
    Prim(Math(op, ty), [lhs, rhs], bind, rest) => {
      let lhs_emit = self.emit_val(lhs)
      let rhs_emit = self.emit_val(rhs)
      let rhs = match (op, ty) {
        (Add, Int) => I32Add(lhs_emit, rhs_emit)
        (Sub, Int) => I32Sub(lhs_emit, rhs_emit)
        (Mul, Int) => I32Mul(lhs_emit, rhs_emit)
        (Div, Int) => I32DivS(lhs_emit, rhs_emit)
        (Add, Double) => F64Add(lhs_emit, rhs_emit)
        (Sub, Double) => F64Sub(lhs_emit, rhs_emit)
        (Mul, Double) => F64Mul(lhs_emit, rhs_emit)
        (Div, Double) => F64Div(lhs_emit, rhs_emit)
      }
      let w_ty = if ty == Int { I32 } else { F64 }
      output.push(LocalSet(bind.to_string(), w_ty, rhs))
      continue rest
    }
    Prim(Eq, [lhs, rhs], bind, rest) => {
      let rhs = match lhs.get_type() {
        Double => F64Eq(self.emit_val(lhs), self.emit_val(rhs))
        Unit => I32Lit(1)
        _ => I32Eq(self.emit_val(lhs), self.emit_val(rhs))
      }
      output.push(LocalSet(bind.to_string(), I32, rhs))
      continue rest
    }
    Prim(Le, [lhs, rhs], bind, rest) => {
      let rhs = match lhs.get_type() {
        Double => F64Le(self.emit_val(lhs), self.emit_val(rhs))
        Unit => I32Lit(1)
        _ => I32LeS(self.emit_val(lhs), self.emit_val(rhs))
      }
      output.push(LocalSet(bind.to_string(), I32, rhs))
      continue rest
    }
    Prim(IntOfFloat, [f], bind, rest) => {
      output.push(
        LocalSet(bind.to_string(), I32, I32TruncSF64(self.emit_val(f))),
      )
      continue rest
    }
    Prim(FloatOfInt, [i], bind, rest) => {
      output.push(
        LocalSet(bind.to_string(), F64, F64ConvertSI32(self.emit_val(i))),
      )
      continue rest
    }
    Prim(AbsFloat, [f], bind, rest) => {
      output.push(LocalSet(bind.to_string(), F64, F64Abs(self.emit_val(f))))
      continue rest
    }
    Prim(_) as expr => @util.die("malformed prim call \{expr}")
    MakeArray(len, elem, kont_closure) => {
      let len_emit = self.emit_val(len)
      let elem_emit = self.emit_val(elem)
      // 1. call to allocate the array
      let make_array_call = match elem.get_type() {
        Double => {
          self.untabled_externals.insert("create_float_array")
          Call("minimbt_create_float_array", [len_emit, elem_emit])
        }
        Unit | Int | Bool => {
          self.untabled_externals.insert("create_array")
          Call("minimbt_create_array", [len_emit, elem_emit])
        }
        _ => {
          self.untabled_externals.insert("create_ptr_array")
          Call("minimbt_create_ptr_array", [len_emit, elem_emit])
        }
      }
      let array_ty = @typing.Type::Array(elem.get_type())
      let array_ret = self.new_named("array", ty=array_ty)
      output.push(LocalSet(array_ret.to_string(), I32, make_array_call))
      // 2. get the continuation address
      let fn_addr = self.new_named("fn_addr", ty=Int)
      self.emit_kth_tuple(0, kont_closure, fn_addr, output)
      // 3. generate the cal to continuation
      self.emit_call(Var(fn_addr), [Var(array_ret), kont_closure], output)
    }
    Exit => output.push(I32Lit(0))
    App(f, args) => self.emit_call(f, args, output)
  }
}

let local_entrance = "minimbt_main"

fn generate_name(name_var : @cps.Var) -> String {
  let referred_by_entrance = name_var.id == 0
  if referred_by_entrance {
    local_entrance
  } else {
    name_var.to_string()
  }
}

pub fn WasmEmitter::emit(self : WasmEmitter) -> WasmTarget {
  let funcs : Array[WasmFunc] = []

  // 1. emit all functions
  for item in self.clops.fnblocks {
    let (label, cur_fn) = item
    let body = []
    self.emit_cps(cur_fn.body, body)
    let params = cur_fn.args.map(
      fn(arg) {
        let ty = match arg.ty.val {
          Double => F64
          Unit => Void
          _ => I32
        }
        let arg_name = arg.to_string()
        (ty, arg_name)
      },
    )
    let name = generate_name(label)
    if params.fold(init=0, fn(acc, ele) { acc + ele.0.get_byte_size() }) >
      params_space_limit {
      @util.die("too few space for storing params for function \{name}")
    }
    funcs.push(
      {
        body,
        name,
        params,
        is_entrance: false,
        closure_name: Some(cur_fn.closure.to_string()),
        func_id: self.function_id[label],
      },
    )
  }
  let body_local_entrance = []
  self.emit_cps(self.clops.root, body_local_entrance)
  funcs.push(
    {
      body: body_local_entrance,
      name: local_entrance,
      params: [],
      is_entrance: false,
      closure_name: None,
      func_id: None,
    },
  )

  // 2. Construct target object
  let function_id = self.function_id
    .iter()
    .map(
      fn {
        (k, v) => (generate_name(self.clops.label_to_closure[k].unwrap()), v)
      },
    )
    |> @hashmap.from_iter()
  let function_id_items = self.function_id.iter().collect()
  function_id_items.sort_by_key(fn { (_, id) => id })
  let fn_table = function_id_items.map(fn { (name, _) => generate_name(name) })
  let closure_id = @hashmap.new()
  let const_fn_arr = self.clops.func_no_free_vars.iter().collect()
  let mut id = 0
  for const_fn in const_fn_arr {
    let const_closure = self.clops.label_to_closure[const_fn].unwrap()
    closure_id[generate_name(const_closure)] = id
    id += 1
  }
  {
    funcs,
    fn_table,
    externals: self.used_externals,
    closure_id,
    function_id,
    untabled_externals: self.untabled_externals,
  }
}
