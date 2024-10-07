pub enum Value {
  Unit
  Int(Int)
  Double(Double)
  Tuple(Array[Value])
  Label(Var)
  Array(Array[Value])
  ExternFn(String)
} derive(Show)

struct CloPSInterpreter {
  clops : @closureps.ClosurePS
  extern_fns : Map[String, (Int, (Array[Value]) -> Value)]
  mut cur_env : @hashmap.T[Var, Value]
}

pub typealias Var = @cps.Var

pub typealias S = @cps.Cps

pub fn Value::op_equal(self : Value, other : Value) -> Bool {
  match (self, other) {
    (Unit, Unit) => true
    (Int(x), Int(y)) => x == y
    (Double(x), Double(y)) => x == y
    (Tuple(xs), Tuple(ys)) => xs == ys
    (Array(xs), Array(ys)) => xs == ys
    (ExternFn(x), ExternFn(y)) => x == y
    _ => false
  }
}

pub type! EvalError {
  VariableNotFind(Var)
} derive(Show)

pub fn CloPSInterpreter::new(clops : @closureps.ClosurePS) -> CloPSInterpreter {
  { extern_fns: Map::new(), clops, cur_env: @hashmap.new() }
}

fn CloPSInterpreter::replace_env(
  self : CloPSInterpreter,
  new_env : @hashmap.T[Var, Value]
) -> CloPSInterpreter {
  { ..self, cur_env: new_env }
}

pub fn CloPSInterpreter::add_extern_fn(
  self : CloPSInterpreter,
  name : String,
  f : (Array[Value]) -> Value,
  argc : Int
) -> Unit {
  self.extern_fns.set(name, (argc, f))
}

fn CloPSInterpreter::find(self : CloPSInterpreter, v : Var) -> Value!EvalError {
  match self.cur_env[v] {
    Some(val) => val
    None =>
      match v.name.val {
        None => raise VariableNotFind(v)
        Some(name) =>
          if self.extern_fns.contains(name) && v.id == -1 {
            ExternFn(name)
          } else {
            raise VariableNotFind(v)
          }
      }
  }
}

fn CloPSInterpreter::eval_v(
  self : CloPSInterpreter,
  value : @cps.Value
) -> Value!EvalError {
  match value {
    Var(v) => self.find!(v)
    Label(v) => Label(v)
    Unit => Unit
    Int(i) => Int(i)
    Double(f) => Double(f)
  }
}

pub fn CloPSInterpreter::eval(
  self : CloPSInterpreter,
  expr : S
) -> Value!EvalError {
  loop expr {
    Exit => break Unit
    Tuple(vs, bind, rest) => {
      let to_binds = []
      for v in vs {
        to_binds.push(self.eval_v!(v))
      }
      self.cur_env[bind] = Tuple(to_binds)
      continue rest
    }
    KthTuple(idx, val, bind, rest) => {
      match self.eval_v!(val) {
        Tuple(tup) => self.cur_env[bind] = tup[idx]
        _ => @util.die("extrating members from non tuple")
      }
      continue rest
    }
    Fix(f, _, _, _) =>
      @util.die("Unexpected non-top level function definiton \{f}")
    Switch(val, branches) =>
      match self.eval_v!(val) {
        Int(idx) => continue branches[idx]
        _ => @util.die("branching on non ints")
      }
    Prim(Not, [v], bind, rest) => {
      match self.eval_v!(v) {
        Int(1) => self.cur_env[bind] = Int(0)
        Int(0) => self.cur_env[bind] = Int(1)
        v => @util.die("unexpected input \{v} for `not`")
      }
      continue rest
    }
    Prim(MakeArray, [len, elem], bind, rest) => {
      match (self.eval_v!(len), self.eval_v!(elem)) {
        (Int(len), elem) => self.cur_env[bind] = Array(Array::make(len, elem))
        (l, elem) => @util.die("unexpected input \{l}, \{elem} for `makearray`")
      }
      continue rest
    }
    Prim(Neg(Double), [f], bind, rest) => {
      match self.eval_v!(f) {
        Double(f) => self.cur_env[bind] = Double(-f)
        v => @util.die("unexpected input \{v} for `neg_double`")
      }
      continue rest
    }
    Prim(Neg(Int), [i], bind, rest) => {
      match self.eval_v!(i) {
        Int(i) => self.cur_env[bind] = Int(-i)
        v => @util.die("unexpected input \{v} for `neg_int`")
      }
      continue rest
    }
    Prim(Get, [arr, idx], bind, rest) => {
      match (self.eval_v!(arr), self.eval_v!(idx)) {
        (Array(arr), Int(idx)) => self.cur_env[bind] = arr[idx]
        (arr, idx) => @util.die("unexpected input \{arr}, \{idx} for `get`")
      }
      continue rest
    }
    Prim(Put, [arr, idx, rhs], bind, rest) => {
      match (self.eval_v!(arr), self.eval_v!(idx), self.eval_v!(rhs)) {
        (Array(arr), Int(idx), rhs) => {
          arr[idx] = rhs
          self.cur_env[bind] = Unit
        }
        (arr, idx, rhs) =>
          @util.die("unexpected input \{arr}, \{idx} and \{rhs} for `put`")
      }
      continue rest
    }
    Prim(Math(op, Int), [lhs, rhs], bind, rest) => {
      match (self.eval_v!(lhs), self.eval_v!(rhs)) {
        (Int(a), Int(b)) => {
          let result = match op {
            Add => Int(a + b)
            Sub => Int(a - b)
            Mul => Int(a * b)
            Div => Int(a / b)
          }
          self.cur_env[bind] = result
        }
        (lhs, rhs) =>
          @util.die("unexpected input \{lhs}, \{rhs} for `\{op}_int`")
      }
      continue rest
    }
    Prim(Math(op, Double), [lhs, rhs], bind, rest) => {
      match (self.eval_v!(lhs), self.eval_v!(rhs)) {
        (Double(a), Double(b)) => {
          let result = match op {
            Add => Double(a + b)
            Sub => Double(a - b)
            Mul => Double(a * b)
            Div => Double(a / b)
          }
          self.cur_env[bind] = result
        }
        (lhs, rhs) =>
          @util.die("unexpected input \{lhs}, \{rhs} for `\{op}_double`")
      }
      continue rest
    }
    Prim(Eq, [lhs, rhs], bind, rest) => {
      let lhs = self.eval_v!(lhs)
      let rhs = self.eval_v!(rhs)
      self.cur_env[bind] = Int(if lhs == rhs { 1 } else { 0 })
      continue rest
    }
    Prim(Le, [lhs, rhs], bind, rest) => {
      match (self.eval_v!(lhs), self.eval_v!(rhs)) {
        (Double(a), Double(b)) =>
          self.cur_env[bind] = Int(if a <= b { 1 } else { 0 })
        (Int(a), Int(b)) => self.cur_env[bind] = Int(if a <= b { 1 } else { 0 })
        (lhs, rhs) => @util.die("unexpected input \{lhs}, \{rhs} for `le`")
      }
      continue rest
    }
    Prim(_) => @util.die("malformed prim call \{expr}")
    App(f, args) => {
      let mut f_val = self.eval_v!(f)
      let mut args_evaled = []
      for arg in args {
        args_evaled.push(self.eval_v!(arg))
      }
      while true {
        guard let Label(address) = f_val else {
          v => @util.die("jumping to non function \{v}")
        }
        match self.clops.fnblocks[address] {
          None => {
            let name = address.name.val.unwrap()
            let (args_needed, extern_fn) = self.extern_fns[name].unwrap()
            // NOTE: There's 2 cases:
            // 1. args_needed + 1 = args_passed, then last arg is continuation
            // 2. args_needed + 2 = args_passed, then 2nd-to-last arg is continuation
            let return_val = extern_fn(args_evaled)
            let cont = args_evaled[args_needed] // rememeber index starts from 0
            args_evaled = [return_val, cont]
            guard let Tuple([fn_ptr, _]) = cont else {
              v => @util.die("\{v} is not a closure")
            }
            f_val = fn_ptr
          }
          _ => break
        }
      }
      guard let Label(address) = f_val else { _ => @util.die("unreachable") }
      let f = self.clops.fnblocks[address].unwrap()
      let new_env : @hashmap.T[Var, Value] = @hashmap.new()
      //let args_to_pass = f.args.copy()
      //args_to_pass.push(f.closure_ref)
      zip2(f.args, args_evaled).each(fn { (k, v) => new_env[k] = v })
      self.cur_env = new_env
      continue f.body
    }
  }
}
