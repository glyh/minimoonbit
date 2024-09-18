pub enum Syntax {
  Unit
  Bool(Bool) // true false
  Int(Int) // int
  Double(Double) // double
  Var(String) // _
  Tuple(Array[Syntax]) // (_, _, _)
  Not(Syntax) // not(_)
  Array(Syntax, Syntax) // Array::make(_, _)
  Neg(Syntax, mut ~kind : Type) // -_
  App(Syntax, Array[Syntax]) // _(_, _, _) or _()
  Get(Syntax, Syntax) // _[_]
  If(Syntax, Syntax, Syntax) // if _ { _ } else { _ }
  Prim(Syntax, Syntax, Op, mut ~kind : Type) // _+ - * / _
  Eq(Syntax, Syntax) // _==_
  LE(Syntax, Syntax) // _<=_
  Let((String, Type), Syntax, Syntax) // let _: _ = _; _
  LetRec(Fundef, Syntax) // fn f() {} ; _
  LetTuple(Array[(String, Type)], Syntax, Syntax) // let (_ , _) : (_, _)= _; _
  Put(Syntax, Syntax, Syntax) // _[_] = _
} derive(Show)

pub enum Op {
  Add
  Sub
  Mul
  Div
} derive(Show, Eq)

pub struct Fundef {
  name : (String, Type)
  args : Array[(String, Type)]
  body : Syntax
} derive(Show)

pub enum Type {
  Unit
  Bool
  Int
  Double
  Ptr
  Fun(Array[Type], Type) // (_, _, _) -> _
  Tuple(Array[Type]) // (_, _, _)
  Array(Type) // Array[_]
  Var(Ref[Result[Type, TVar]])
} derive(Show)

pub fn Type::is_ptr_like(self : Type) -> Bool {
  match self {
    Fun(_) | Tuple(_) | Array(_) | Ptr => true
    Var(t) =>
      match t.val {
        Ok(t) => t.is_ptr_like()
        Err(_) => false
      }
    _ => false
  }
}

pub fn Type::is_float_like(self : Type) -> Bool {
  match self {
    Double => true
    Var(t) =>
      match t.val {
        Ok(t) => t.is_float_like()
        Err(_) => false
      }
    _ => false
  }
}

pub fn Type::size_of(self : Type, size_of_ptr : Int) -> Int {
  match self {
    Unit => 0
    Bool => 4
    Int => 4
    Double => 8
    Tuple(_) | Fun(_, _) | Array(_) | Ptr => size_of_ptr
    Var(t) =>
      match t.val {
        Ok(t) => t.size_of(size_of_ptr)
        Err(_) => @util.die("Uninstantiated type variable")
      }
  }
}

pub fn Type::op_equal(self : Type, other : Type) -> Bool {
  match (self, other) {
    (Unit, Unit) => true
    (Bool, Bool) => true
    (Int, Int) => true
    (Double, Double) => true
    (Fun(xs, x), Fun(ys, y)) => xs == ys && x == y
    (Tuple(xs), Tuple(ys)) => xs == ys
    (Array(x), Array(y)) => x == y
    (Ptr, Ptr) => true
    (Var({ val: Ok(x) }), y) | (x, Var({ val: Ok(y) })) => x == y
    (Var({ val: Err(x) }), Var({ val: Err(y) })) => x == y
    _ => false
  }
}

pub fn Type::is_known(self : Type) -> Bool {
  fn all_known(arr : Array[Type]) {
    arr.fold(init=true, fn(acc : Bool, ele : Type) { acc && ele.is_known() })
  }

  match self {
    Unit | Bool | Int | Double | Ptr => true
    Fun(args, ret) => all_known(args) && ret.is_known()
    Tuple(tup) => all_known(tup)
    Array(t) => t.is_known()
    Var({ val: Ok(t) }) => t.is_known()
    Var({ val: Err(_) }) => false
  }
}

pub fn Type::normalize(self : Type) -> Type {
  match self {
    Unit | Bool | Int | Double | Ptr => self
    Fun(args, ret) => Fun(args.map(normalize), ret.normalize())
    Tuple(tup) => Tuple(tup.map(normalize))
    Array(t) => Array(t.normalize())
    Var({ val: Ok(t) }) => t.normalize()
    Var({ val: Err(_) }) => self
  }
}
