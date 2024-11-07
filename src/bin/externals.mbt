fn add_clops_interp_fns(interpreter : @closureps_eval.CloPSInterpreter) -> Unit {
  interpreter.add_extern_fn(
    "print_char",
    fn(args) {
      match args[0] {
        Int(i) => @io.print_char(Char::from_int(i))
        _ => @util.die("print_char expects Int")
      }
      Unit
    },
    1,
  )
  interpreter.add_extern_fn(
    "print_int",
    fn(args) {
      match args[0] {
        Int(i) => @io.print(i)
        _ => @util.die("print_int expects Int")
      }
      Unit
    },
    1,
  )
  interpreter.add_extern_fn(
    "print_endline",
    fn(_args) {
      @io.print("\n")
      Unit
    },
    0,
  )
  let create_array = fn(args : Array[@closureps_eval.Value]) {
    match args[0] {
      Int(n) => @closureps_eval.Value::Array(Array::make(n, args[1]))
      _ => @util.die("create_array expects Int")
    }
  }
  interpreter.add_extern_fn("create_array", create_array, 2)
  interpreter.add_extern_fn("create_float_array", create_array, 2)
  interpreter.add_extern_fn("create_ptr_array", create_array, 2)
  interpreter.add_extern_fn(
    "truncate",
    fn(args) {
      match args[0] {
        Double(d) => Int(d.to_int())
        _ => @util.die("expects Double")
      }
    },
    1,
  )
  interpreter.add_extern_fn(
    "sin",
    fn(args) {
      match args[0] {
        Double(d) => Double(@math.sin(d))
        _ => @util.die("sin expects Double")
      }
    },
    1,
  )
  interpreter.add_extern_fn(
    "cos",
    fn(args) {
      match args[0] {
        Double(d) => Double(@math.cos(d))
        _ => @util.die("cos expects Double")
      }
    },
    1,
  )
  interpreter.add_extern_fn(
    "sqrt",
    fn(args) {
      match args[0] {
        Double(d) => Double(d.sqrt())
        _ => @util.die("sqrt expects Double")
      }
    },
    1,
  )
  interpreter.add_extern_fn(
    "abs_float",
    fn(args) {
      match args[0] {
        Double(d) => Double(@double.abs(d))
        _ => @util.die("abs_float expects Double")
      }
    },
    1,
  )
  interpreter.add_extern_fn(
    "int_of_float",
    fn(args) {
      match args[0] {
        Double(d) => Int(d.to_int())
        _ => @util.die("int_of_float expects Double")
      }
    },
    1,
  )
  interpreter.add_extern_fn(
    "float_of_int",
    fn(args) {
      match args[0] {
        Int(i) => Double(i.to_double())
        _ => @util.die("float_of_int expects Int")
      }
    },
    1,
  )
}
