/// Low-level types. Main difference between this and Type is that it distinguishes between direct function pointers and closure function pointers.
pub enum LowType {
  Unit
  Int
  Double
  // true indicates this is a closure, while false indicate otherwise
  Fn(Array[LowType], LowType, Ref[Bool])
  Tuple(Array[LowType])
  Array(LowType)
  Ptr
} derive(Show)

pub fn LowType::is_ptr_like(self : LowType) -> Bool {
  match self {
    Fn(_) | Tuple(_) | Array(_) | Ptr(_) => true
    _ => false
  }
}

pub fn LowType::is_float_like(self : LowType) -> Bool {
  match self {
    Double => true
    _ => false
  }
}

pub fn LowType::size_of(self : LowType, size_of_ptr : Int) -> Int {
  match self {
    Unit => 0
    Int => 4
    Double => 8
    _ => if self.is_ptr_like() { size_of_ptr } else { 4 }
  }
}

pub fn LowType::of_ty_gen(t : Type) -> LowType {
  match t {
    Unit => Unit
    Bool | Int => Int
    Double => Double
    Ptr => Ptr
    Fun(args, ret) => Fn(args.map(of_ty_gen), of_ty_gen(ret), { val: true })
    Tuple(tys) => Tuple(tys.map(of_ty_gen))
    Array(ty) => Array(of_ty_gen(ty))
    Var(_) => @util.die("Converting uninferred type to low type")
  }
}

pub fn LowType::of_ty(t : Type) -> LowType {
  match t {
    Unit => Unit
    Bool | Int => Int
    Double => Double
    Ptr => Ptr
    Fun(args, ret) => Fn(args.map(of_ty), of_ty(ret), { val: false })
    Tuple(tys) => Tuple(tys.map(of_ty))
    Array(ty) => Array(of_ty(ty))
    Var(_) => @util.die("Converting uninferred type to low type")
  }
}

pub fn LowType::mark_as_closure(self : LowType) -> Unit {
  match self {
    Fn(_, _, is_closure) => is_closure.val = true
    _ => @util.die("Conversion to closure expects \{self} to be callable")
  }
}

pub fn LowType::op_equal(self : LowType, other : LowType) -> Bool {
  match (self, other) {
    (Unit, Unit) | (Int, Int) | (Double, Double) | (Ptr, Ptr) => true
    (Array(lhs), Array(rhs)) => lhs == rhs
    (Tuple(lhss), Tuple(rhss)) =>
      lhss.length() == rhss.length() &&
      (zip2(lhss, rhss) |> forall(fn { (lhs, rhs) => lhs == rhs }))
    (Fn(args1, ret1, clo1), Fn(args2, ret2, clo2)) =>
      clo1.val == clo2.val &&
      args1.length() == args2.length() &&
      (zip2(args1, args2) |> forall(fn { (lhs, rhs) => lhs == rhs })) &&
      ret1 == ret2
    _ => false
  }
}

pub fn LowType::unify_clo(self : LowType, other : LowType) -> Unit {
  match (self, other) {
    (Unit, Unit) | (Int, Int) | (Double, Double) | (Ptr, Ptr) => ()
    (Array(lhs), Array(rhs)) => lhs.unify_clo(rhs)
    (Tuple(lhss), Tuple(rhss)) => {
      if lhss.length() != rhss.length() {
        @util.die("Expect \{self} to have same structure with \{other}")
      }
      zip2(lhss, rhss).each(fn { (lhs, rhs) => lhs.unify_clo(rhs) })
    }
    (Fn(args1, ret1, clo1), Fn(args2, ret2, clo2)) => {
      if args1.length() != args2.length() {
        @util.die("Expect \{self} to have same structure with \{other}")
      }
      zip2(args1, args2).each(fn { (lhs, rhs) => lhs.unify_clo(rhs) })
      ret1.unify_clo(ret2)
      if clo1.val || clo2.val {
        clo1.val = true
        clo2.val = true
      }
    }
    _ => @util.die("Can't unify between \{self} and \{other}")
  }
}
