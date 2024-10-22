// NOTE: 
// PreCPS IR different from typed AST in that:
// 1. all expression carries a type, or can be have it's type inferred without 
// an associated type enviornment 
// 2. All variables that refers to the same object has a same unique ID.
pub enum PreCps {
  Unit
  Int(Int)
  Double(Double)
  Bool(Bool)
  // T marks the binding type
  LetRec(T, Var, Array[Var], PreCps, PreCps)
  Let(T, Var, PreCps, PreCps)
  // T marks return type
  Var(T, Var)
  Label(T, Var)
  Tuple(T, Array[PreCps])
  Prim(T, PrimOp, Array[PreCps])
  KthTuple(T, Int, PreCps)
  App(T, PreCps, Array[PreCps])
  If(T, PreCps, PreCps, PreCps)
}

impl Show for PreCps with output(self, logger) {
  logger.write_string(self.to_string())
}

pub fn PreCps::to_string(self : PreCps) -> String {
  match self {
    Unit => "()"
    Int(i) => i.to_string()
    Bool(b) => b.to_string()
    Double(f) => f.to_string()
    LetRec(_, name, args, body, rest) =>
      "letrec \{name}(\{args}){\{body}} in \{rest}"
    Let(_, name, rhs, rest) => "let \{name} = \{rhs} in \{rest}"
    Var(_, v) => v.to_string()
    Label(_, v) => ":" + v.to_string()
    Tuple(_, tup) => tup.to_string()
    KthTuple(_, idx, tup) => "(\{tup}).\{idx}"
    App(_, f, args) => "(\{f} \{args})"
    Prim(_, op, args) => "#(\{op} \{args})"
    If(_, cond, _then, _else) =>
      "(if { \{cond} } then { \{_then} } else { \{_else}) }"
  }
}

pub fn PreCps::get_type(self : PreCps) -> T {
  match self {
    Unit => Unit
    Int(_) => Int
    Double(_) => Double
    Bool(_) => Bool
    LetRec(_, _, _, _, inner) => inner.get_type()
    Let(_, _, _, inner) => inner.get_type()
    Var(t, _)
    | Label(t, _)
    | Tuple(t, _)
    | Prim(t, _, _) | KthTuple(t, _, _) | App(t, _, _) | If(t, _, _, _) => t
  }
}

pub enum Numeric {
  Double
  Int
} derive(Show, Eq, Hash)

pub enum PrimOp {
  Not
  MakeArray
  Neg(Numeric)
  Get
  Put
  Math(@types.Op, Numeric)
  Eq
  Le
} derive(Show)
