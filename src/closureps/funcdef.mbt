// we don't store the return value type as there's no return in CPS
pub struct FuncDef {
  args : Array[Var]
  free_vars : Array[Var]
  // closure is a tuple of function pointer
  // and a tuple of free variables
  body : S
}
