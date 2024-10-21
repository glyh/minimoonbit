fn SsaCfg::cps2block(
  self : SsaCfg,
  s : Cps,
  cur_block : Block,
  cur_label : Var
) -> Unit {
  let mut cur_block = cur_block
  let mut cur_label = cur_label
  loop s {
    Tuple(record, bind, rest) => {
      cur_block.insts.push(MakeTuple(bind, record))
      continue rest
    }
    KthTuple(idx, tup, bind, rest) => {
      cur_block.insts.push(KthTuple(bind, tup, idx))
      continue rest
    }
    Prim(Not, args, bind, rest) => {
      cur_block.insts.push(Prim(bind, Not, args))
      continue rest
    }
    Prim(Neg(n), args, bind, rest) => {
      cur_block.insts.push(Prim(bind, Neg(n), args))
      continue rest
    }
    Prim(Get, args, bind, rest) => {
      cur_block.insts.push(Prim(bind, Get, args))
      continue rest
    }
    Prim(Math(op, n), args, bind, rest) => {
      cur_block.insts.push(Prim(bind, Math(op, n), args))
      continue rest
    }
    Prim(Eq, args, bind, rest) => {
      cur_block.insts.push(Prim(bind, Eq, args))
      continue rest
    }
    Prim(Le, args, bind, rest) => {
      cur_block.insts.push(Prim(bind, Le, args))
      continue rest
    }
    // NOTE: we have to break store into 2 3AC because we need 2 tmps
    Prim(Put, [arr, idx, rhs], bind, rest) => {
      let ptr = self.new_named("element_ptr", ty=@typing.Type::Ptr)
      cur_block.insts.append(
        [
          Prim(ptr, GetArrPtr, [arr, idx]),
          Prim(bind, StoreToPtr, [rhs, Var(ptr)]),
        ],
      )
      continue rest
    }
    Prim(Put, args, _, _) => @util.die("unexpect args \{args} for put")
    Fix(_) => @util.die("unexpected nested function")
    If(cond, _then, _else) => {
      // NOTE: for control flow connection points there's no reason for us to 
      // generate a type for it.
      // NOTE: complete cur block; phis need to be fixed, but we'll deal with it in a later pass 
      let label_then = self.new_named("then")
      let label_else = self.new_named("else")
      cur_block.last_inst.val = Branch(cond, label_then, label_else)
      // NOTE: convert then block to ssa block 
      let block_then = Block::new_from(cur_block.fn_name, cur_label)
      self.blocks[label_then] = block_then
      self.cps2block(_then, block_then, label_then)
      // NOTE: convert else block to ssa block 
      let block_else = Block::new_from(cur_block.fn_name, cur_label)
      self.blocks[label_else] = block_else
      cur_block = block_else
      cur_label = label_else
      continue _else
      // no need to add merge point after if statements as we always have a tail call
    }
    App(f, args) => {
      cur_block.last_inst.val = Call(f, args)
      break
    }
    MakeArray(len, elem, kont) => {
      cur_block.last_inst.val = MakeArray(len, elem, kont)
      break
    }
    Exit => {
      cur_block.last_inst.val = Exit
      break
    }
  }
}
