pub enum Inst {
  // cps ir
  MakeTuple(Var, Array[Value])
  KthTuple(Var, Value, Int)
  // primitives
  Prim(Var, @cps.PrimOp, Array[Value])
  // for reg allocation spilling
  Store(Var)
  Load(Var)
} derive(Show)

pub enum PCInst {
  // join points (val, block ref)
  Branch(Value, Var, Var)
  Call(Value, Array[Value])
  // len, elem, continuation
  MakeArray(Value, Value, Value)
  Exit
} derive(Show)

pub struct Block {
  from : Var?
  fn_name : Var // for TCO
  insts : Array[Inst]
  last_inst : Ref[PCInst]
} derive(Show)

fn Block::new_from(fn_name : Var, from : Var) -> Block {
  { fn_name, from: Some(from), insts: [], last_inst: { val: Exit } }
}

pub struct SsaCfg {
  blocks : @hashmap.T[Var, Block]
  fn_args : @hashmap.T[Var, Array[Var]]
  spilled : @hashmap.T[Var, @hashset.T[Var]]
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

fn SsaCfg::new_named(self : SsaCfg, name : String) -> Var {
  self.counter += 1
  // NOTE: for control flow connection poitns there's no reason for us to 
  // generate a type for it.
  { name: { val: Some(name) }, id: self.counter, ty: { val: Unit } }
}
