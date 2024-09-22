pub typealias Name = @types.Name

pub typealias Type = @types.Type

pub enum Expr {
  Unit
  Int(Int)
  Double(Double)
  Var(Name)
  MakeTuple(Array[Name])
  Neg(Name)
  CallClosure(Name, Array[Name])
  CallDirect(Label, Array[Name])
  ArrayGet(Name, Name)
  IfEq(Name, Name, Expr, Expr)
  IfLe(Name, Name, Expr, Expr)
  Add(Name, Name)
  Sub(Name, Name)
  Mul(Name, Name)
  Div(Name, Name)
  FNeg(Name)
  FAdd(Name, Name)
  FSub(Name, Name)
  FMul(Name, Name)
  FDiv(Name, Name)
  Let((Name, LowType), Expr, Expr)
  MakeClosure((Name, LowType), Closure, Expr)
  LetTuple(Array[(Name, LowType)], Name, Expr)
  ArrayPut(Name, Name, Name)
  ExternalArray(Label)
} derive(Show)

pub struct Closure {
  name : Label
  actual_free_vars : Array[Name]
} derive(Show)

pub struct FuncDef {
  name : Label
  old_name : Name
  /// true if the function is a closure function, and the closure will be available at `s11` on 
  /// function entry.
  is_closure : Bool
  ty : LowType
  args : Array[(Name, LowType)]
  formal_free_vars : Array[(Name, LowType)]
  mut body : Expr
} derive(Show)

pub struct Program {
  fundefs : Array[FuncDef]
  body : Expr
} derive(Show)
