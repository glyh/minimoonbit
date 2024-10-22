pub enum PrimOp {
  Not
  Neg(@precps.Numeric)
  Get
  GetArrPtr
  StoreToPtr
  Math(@typing.Op, @precps.Numeric)
  Eq
  Le
} derive(Show)

pub enum Inst {
  // Cps IR
  MakeTuple(Var, Array[Value])
  KthTuple(Var, Value, Int)
  // Primitives
  Prim(Var, PrimOp, Array[Value])
  // For reg allocation spilling
  Store(Var)
  Load(Var)
  // So we can deal with the case where 1 tmp reg is not enough
  Copy(Var, Value)
} derive(Show)

pub enum PCInst {
  // join points (val, block ref)
  Branch(Value, Var, Var)
  BranchEq(Value, Value, Var, Var)
  BranchLe(Value, Value, Var, Var)
  Call(Value, Array[Value])
  // len, elem, continuation
  MakeArray(Value, Value, Value)
  Exit
} derive(Show)

pub struct Block {
  from : Var?
  fn_name : Var // for TCO
  mut insts : Array[Inst] //
  last_inst : Ref[PCInst]
} derive(Show)

fn Block::new_from(fn_name : Var, from : Var) -> Block {
  { fn_name, from: Some(from), insts: [], last_inst: { val: Exit } }
}

pub struct SsaCfg {
  blocks : @hashmap.T[Var, Block]
  fn_args : @hashmap.T[Var, Array[Var]]
  spilled : @hashmap.T[Var, @hashset.T[Var]]
  label_to_closure : @hashmap.T[Var, Var]
  func_no_free_vars : @hashset.T[Var]
  mut counter : Int
}

fn SsaCfg::new_entry(self : SsaCfg, entry_name : Var) -> Block {
  self.fn_args[entry_name] = []
  { fn_name: entry_name, from: None, insts: [], last_inst: { val: Exit } }
}

fn SsaCfg::new_fn_block(
  self : SsaCfg,
  fn_name : Var,
  func_block : @closureps.FuncDef
) -> Block {
  let args_with_closure = func_block.args.copy()
  args_with_closure.push(func_block.closure)
  self.fn_args[fn_name] = args_with_closure
  { fn_name, from: None, insts: [], last_inst: { val: Exit } }
}

pub fn SsaCfg::new_named(
  self : SsaCfg,
  name : String,
  ~ty : @typing.Type = Unit
) -> Var {
  self.counter += 1
  { name: { val: Some(name) }, id: self.counter, ty: { val: ty } }
}
