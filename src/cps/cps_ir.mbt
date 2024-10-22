pub struct Var {
  name : Ref[String?]
  id : Int
  ty : Ref[T]
}

pub fn Var::compare(lhs : Var, rhs : Var) -> Int {
  lhs.id.compare(rhs.id)
}

pub fn Var::hash_combine(self : Var, hasher : Hasher) -> Unit {
  hasher.combine(self.id)
}

pub fn Var::op_equal(lhs : Var, rhs : Var) -> Bool {
  lhs.id == rhs.id
}

fn Var::from_precps(v : @precps.Var, t : T) -> Var {
  { id: v.id, name: { val: v.name }, ty: { val: t } }
}

pub enum Value {
  Var(Var)
  Label(Var)
  Unit
  Int(Int)
  Bool(Bool)
  Double(Double)
} derive(Eq)

fn Value::replace_var_bind(self : Value, from : Var, to : Value) -> Value {
  match self {
    Var(v) => if v == from { to } else { self }
    _ => self
  }
}

pub fn Value::get_type(self : Value) -> T {
  match self {
    Var(var) | Label(var) => var.ty.val
    Unit => Unit
    Int(_) => Int
    Double(_) => Double
    Bool(_) => Bool
  }
}

pub enum PrimOp {
  Not
  Neg(@precps.Numeric)
  Get
  Put
  Math(@top.Op, @precps.Numeric)
  Eq
  Le
} derive(Show)

pub enum Cps {
  // T marks the binding's type
  Tuple(Array[Value], Var, Cps)
  KthTuple(Int, Value, Var, Cps)
  Fix(Var, Array[Var], Cps, Cps)
  If(Value, Cps, Cps)
  IfLe(Value, Value, Cps, Cps)
  IfEq(Value, Value, Cps, Cps)
  Prim(PrimOp, Array[Value], Var, Cps)
  // T marks the return type
  App(Value, Array[Value])
  // len, elem, continuation
  MakeArray(Value, Value, Value)
  Exit
}

fn Cps::replace_var_bind(self : Cps, from : Var, to : Value) -> Cps {
  fn rec(s : Cps) {
    s.replace_var_bind(from, to)
  }

  fn recv(v : Value) {
    v.replace_var_bind(from, to)
  }

  match self {
    Tuple(record, bind, rest) => {
      let rest_new = if from != bind { rec(rest) } else { rest }
      Tuple(record.map(recv), bind, rest_new)
    }
    KthTuple(idx, v, bind, rest) => {
      let rest_new = if from != bind { rec(rest) } else { rest }
      KthTuple(idx, recv(v), bind, rest_new)
    }
    Fix(name, args, body, rest) => {
      let body_new = if from != name && not(args.contains(from)) {
        rec(body)
      } else {
        body
      }
      let rest_new = if from != name { rec(rest) } else { body }
      Fix(name, args, body_new, rest_new)
    }
    If(cond, _then, _else) => If(recv(cond), rec(_then), rec(_else))
    IfEq(lhs, rhs, _then, _else) =>
      IfEq(recv(lhs), recv(rhs), rec(_then), rec(_else))
    IfLe(lhs, rhs, _then, _else) =>
      IfLe(recv(lhs), recv(rhs), rec(_then), rec(_else))
    Prim(op, args, bind, rest) => {
      let rest_new = if from != bind { rec(rest) } else { rest }
      Prim(op, args.map(recv), bind, rest_new)
    }
    App(f, args) => App(recv(f), args.map(recv))
    MakeArray(len, elem, cont) => MakeArray(recv(len), recv(elem), recv(cont))
    Exit => Exit
  }
}

fn Value::free_variables(self : Value) -> @hashset.T[Var] {
  match self {
    Var(v) => @hashset.of([v])
    _ => @hashset.new()
  }
}

pub fn Cps::free_variables(self : Cps) -> @hashset.T[Var] {
  match self {
    Tuple(record, bind, rest) => {
      let fvs = rest.free_variables()
      fvs.remove(bind)
      record.fold(init=fvs, fn(acc, ele) { acc.union(ele.free_variables()) })
    }
    KthTuple(_, v, bind, rest) => {
      let fvs = rest.free_variables()
      fvs.remove(bind)
      fvs.union(v.free_variables())
    }
    Fix(name, args, body, rest) => {
      let fv_rest = rest.free_variables()
      fv_rest.remove(name)
      let fv_body = body.free_variables()
      fv_body.remove(name)
      args.each(fn { a => fv_body.remove(a) })
      fv_body.union(fv_rest)
    }
    If(cond, _then, _else) =>
      cond
      .free_variables()
      .union(_then.free_variables())
      .union(_else.free_variables())
    IfEq(lhs, rhs, _then, _else) | IfLe(lhs, rhs, _then, _else) =>
      lhs
      .free_variables()
      .union(rhs.free_variables())
      .union(_then.free_variables())
      .union(_else.free_variables())
    Prim(_, args, bind, rest) => {
      let fv_rest = rest.free_variables()
      fv_rest.remove(bind)
      args.fold(init=fv_rest, fn(acc, ele) { acc.union(ele.free_variables()) })
    }
    App(f, args) =>
      args.fold(
        init=f.free_variables(),
        fn(acc, ele) { acc.union(ele.free_variables()) },
      )
    MakeArray(len, elem, cont) =>
      len
      .free_variables()
      .union(elem.free_variables())
      .union(cont.free_variables())
    Exit => @hashset.new()
  }
}
