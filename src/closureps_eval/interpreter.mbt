// TODO: create base env that includes closure of non closures
pub enum Value {
  Unit
  Int(Int)
  Double(Double)
  Tuple(Array[Value])
  Bool(Bool)
  Label(Var)
  Array(Array[Value])
  ExternFn(String)
} derive(Show)

struct CloPSInterpreter {
  clops : @closureps.ClosurePS
  extern_fns : Map[String, (Int, (Array[Value]) -> Value)]
  mut cur_env : @hashmap.T[Var, Value]
}

// NOTE: no worry of repeated names generated as all vars are marked by an uid
fn CloPSInterpreter::new_named(
  self : CloPSInterpreter,
  name : String,
  t : @types.Type
) -> Var {
  self.clops.counter.val = self.clops.counter.val + 1
  { name: { val: Some(name) }, id: self.clops.counter.val, ty: { val: t } }
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

fn CloPSInterpreter::reset_env(self : CloPSInterpreter) -> Unit {
  self.cur_env = @hashmap.new()
  for f in self.clops.func_no_free_vars {
    let clo = self.clops.label_to_closure[f].unwrap()
    self.cur_env[clo] = Tuple([Label(f), Unit])
  }
}

pub fn CloPSInterpreter::new(clops : @closureps.ClosurePS) -> CloPSInterpreter {
  let ret = { extern_fns: Map::new(), clops, cur_env: @hashmap.new() }
  ret.reset_env()
  ret
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
          if self.extern_fns.contains(name) && v.id < 0 {
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
    Bool(b) => Bool(b)
  }
}

pub fn CloPSInterpreter::call(
  self : CloPSInterpreter,
  f_val : Value,
  args_val : Array[Value],
  closure_ref : Value
) -> S {
  let mut f_val = f_val
  let mut args_val = args_val
  //let mut closure_ref = closure_ref
  while true {
    guard let Label(address) = f_val else {
      v => @util.die("jumping to non function \{v}")
      //println("Cur function: \{f_val}")
    }
    match self.clops.fnblocks[address] {
      None => {
        let name = address.name.val.unwrap()
        let (_, extern_fn) = self.extern_fns[name].unwrap()
        let return_val = extern_fn(args_val)
        let cont = closure_ref // that closure is actually a continuation
        args_val = [return_val]
        //closure_ref = Unit
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
  self.reset_env()
  zip2(f.args, args_val).each(fn { (k, v) => self.cur_env[k] = v })
  self.cur_env[f.closure] = closure_ref
  f.body
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
        result =>
          @util.die("Extracting members from non tuple \{result} @ \{idx}")
      }
      continue rest
    }
    Fix(f, _, _, _) =>
      @util.die("Unexpected non-top level function definiton \{f}")
    If(cond, _then, _else) =>
      match self.eval_v!(cond) {
        Bool(b) => continue if b { _then } else { _else }
        v => @util.die("unexpected condition \{v} for `if`")
      }
    Prim(Not, [v], bind, rest) => {
      match self.eval_v!(v) {
        Bool(b) => self.cur_env[bind] = Bool(not(b))
        v => @util.die("unexpected input \{v} for `not`")
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
      self.cur_env[bind] = Bool(lhs == rhs)
      continue rest
    }
    Prim(Le, [lhs, rhs], bind, rest) => {
      match (self.eval_v!(lhs), self.eval_v!(rhs)) {
        (Double(a), Double(b)) => self.cur_env[bind] = Bool(a <= b)
        (Int(a), Int(b)) => self.cur_env[bind] = Bool(a <= b)
        (lhs, rhs) => @util.die("unexpected input \{lhs}, \{rhs} for `le`")
      }
      continue rest
    }
    Prim(_) => @util.die("malformed prim call \{expr}")
    MakeArray(len, elem, kont_closure) =>
      match (self.eval_v!(len), self.eval_v!(elem)) {
        (Int(len), elem) => {
          let arr = Array(Array::make(len, elem))
          let closure_val = self.eval_v!(kont_closure)
          guard let Tuple([fn_address, _]) = closure_val else {
            _ =>
              @util.die(
                "evaluate to non closure passed in to makearray \{expr}",
              )
          }
          continue self.call(fn_address, [arr], closure_val)
        }
        (l, elem) => @util.die("unexpected input \{l}, \{elem} for `makearray`")
      }
    App(f, args) => {
      let f_val = self.eval_v!(f)
      let args_val = []
      for arg in args {
        args_val.push(self.eval_v!(arg))
      }
      let closure_ref_val = args_val.pop().unwrap()
      continue self.call(f_val, args_val, closure_ref_val)
    }
  }
}
