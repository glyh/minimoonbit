pub typealias Name = @types.Name

pub typealias Type = @types.Type

pub enum Knf {
  Unit
  Int(Int)
  Double(Double)
  Var(Name)
  Tuple(Array[Name])
  Neg(Name)
  Apply(Name, Array[Name])
  Get(Name, Name)
  IfEq(Name, Name, Knf, Knf)
  IfLe(Name, Name, Knf, Knf)
  Add(Name, Name)
  Sub(Name, Name)
  Mul(Name, Name)
  Div(Name, Name)
  FNeg(Name)
  FAdd(Name, Name)
  FSub(Name, Name)
  FMul(Name, Name)
  FDiv(Name, Name)
  Let((Name, Type), Knf, Knf)
  LetRec(FuncDef, Knf)
  LetTuple(Array[(Name, Type)], Name, Knf)
  Put(Name, Name, Name)
  ExternalArray(Name)
  ExternalFunctionApplication(String, Array[Name])
} derive(Show, Eq)

pub struct FuncDef {
  name : Name
  ty : Type
  args : Array[(Name, Type)]
  body : Knf
} derive(Show, Eq)

fn FuncDef::from_syntax(kenv : KnfEnv, f : @types.Fundef) -> FuncDef {
  let { name: (f_id, ty), args, body } = f
  let name = kenv.new_name(f_id)
  let args = args.map(
    fn {
      (id, ty) => {
        let name = kenv.new_name(id)
        (name, ty)
      }
    },
  )
  let fn_env = args.fold(
    init=kenv.add(name, ty),
    fn(env, arg) { env.add(arg.0, arg.1) },
  )
  let (body, _) = to_knf_with_tenv(fn_env, body)
  { name, ty, args, body }
}
