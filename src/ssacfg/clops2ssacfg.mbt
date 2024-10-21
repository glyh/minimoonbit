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
    Prim(op, args, bind, rest) => {
      cur_block.insts.push(Prim(bind, op, args))
      continue rest
    }
    Fix(_) => @util.die("unexpected nested function")
    If(cond, _then, _else) => {
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
