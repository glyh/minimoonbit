pub struct Var {
  name : Ref[String?]
  id : Int
  ty : T
}

pub fn Var::hash_combine(self : Var, hasher : Hasher) -> Unit {
  hasher.combine(self.id)
}

pub fn Var::op_equal(lhs : Var, rhs : Var) -> Bool {
  lhs.id == rhs.id
}

fn Var::from_precps(v : @precps.Var, t : T) -> Var {
  { id: v.id, name: { val: v.name }, ty: t }
}

pub enum Value {
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

pub enum Cps {
  // T marks the binding's type
  Tuple(Array[Value], Var, Cps)
  KthTuple(Int, Value, Var, Cps)
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
    Switch(v, branches) => Switch(recv(v), branches.map(rec))
    Prim(op, args, bind, rest) => {
      let rest_new = if from != bind { rec(rest) } else { rest }
      Prim(op, args.map(recv), bind, rest_new)
    }
    App(f, args) => App(recv(f), args.map(recv))
    Just(v) => Just(recv(v))
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
    Switch(v, branches) =>
      branches.fold(
        init=v.free_variables(),
        fn(acc, ele) { acc.union(ele.free_variables()) },
      )
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
    Just(v) => v.free_variables()
  }
}
