struct Var {
  name : Ref[String?]
  id : Int
  ty : T
}

fn Var::op_equal(lhs : Var, rhs : Var) -> Bool {
  lhs.id == rhs.id
}

fn Var::from_precps(v : @precps.Var, t : T) -> Var {
  { id: v.id, name: { val: v.name }, ty: t }
}

enum Value {
  Var(Var)
  Label(Var)
  Unit
  Int(Int)
  Double(Double)
} derive(Eq)

fn Value::replace_var_bind(self : Value, from : Var, to : Value) -> Value {
  match self {
    Var(v) => if v == from { to } else { self }
    _ => self
  }
}

enum AccessPath {
  OffP(Int)
  SelP(Int, AccessPath)
} derive(Show)

pub enum Cps {
  // T marks the binding's type
  Record(Array[(Value, AccessPath)], Var, Cps)
  Select(Int, Value, Var, Cps)
  Offset(Int, Value, Var, Cps)
  Fix(Var, Array[Var], Cps, Cps)
  Switch(Value, Array[Cps])
  Prim(PrimOp, Array[Value], Var, Cps)
  // T marks the return type
  App(Value, Array[Value])
  Just(Value)
}

fn Cps::replace_var_bind(self : Cps, from : Var, to : Value) -> Cps {
  fn rec(s : Cps) {
    s.replace_var_bind(from, to)
  }

  fn recv(v : Value) {
    v.replace_var_bind(from, to)
  }

  match self {
    Record(record, bind, rest) => {
      let rest_new = if from != bind { rec(rest) } else { rest }
      Record(record.map(fn { (v, path) => (recv(v), path) }), bind, rest_new)
    }
    Select(idx, v, bind, rest) => {
      let rest_new = if from != bind { rec(rest) } else { rest }
      Select(idx, recv(v), bind, rest_new)
    }
    Offset(idx, v, bind, rest) => {
      let rest_new = if from != bind { rec(rest) } else { rest }
      Offset(idx, recv(v), bind, rest_new)
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
    Switch(v, branches) => Switch(recv(v), branches.map(rec))
    Prim(op, args, bind, rest) => {
      let rest_new = if from != bind { rec(rest) } else { rest }
      Prim(op, args.map(recv), bind, rest_new)
    }
    App(f, args) => App(recv(f), args.map(recv))
  }
}
