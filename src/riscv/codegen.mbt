// TODO: 
// 1. aggressively throw free variables of current continuation closure on stored registers
enum RegTy {
  I32
  PTR64
  F64
  Void
} derive(Show, Eq)

fn RegTy::byte_size(r : RegTy) -> Int {
  match r {
    I32 => 32 / 8
    PTR64 | F64 => 64 / 8
    Void => 0
  }
}

fn RegTy::byte_size_bits(r : RegTy) -> Int {
  match r {
    I32 => 2
    PTR64 | F64 => 3
    Void => 0
  }
}

fn ty_get_reg_ty(t : T) -> RegTy {
  match t {
    Double => F64
    Unit => Void
    Bool | Int => I32
    Ptr | Fun(_) | Tuple(_) | Array(_) => PTR64
    Var(_) => @util.die("unreachable: type not unified")
  }
}

fn get_reg_ty(v : Value) -> RegTy {
  match v {
    Var(var) => ty_get_reg_ty(var.ty.val)
    Label(_) => PTR64
    Unit => Void
    Int(_) | Bool(_) => I32
    Double(_) => F64
  }
}

struct CodegenBlock {
  // global infos
  cfg : @ssacfg.SsaCfg
  allocation : @hashmap.T[Var, RegRef]
  cur_fn : Ref[AssemblyFunction]
  // local infos
  block_label : Var
  // mutable states
  mut stack_offset : Int
  mut dirtied : @hashset.T[RegRef]
  spilled_offset : @hashmap.T[Var, Int]
}

let global_entrance = "minimbt_main"

fn generate_name(name_var : Var) -> String {
  let is_entrance = name_var.id == 0
  if is_entrance {
    global_entrance
  } else {
    name_var.to_string()
  }
}

fn CodegenBlock::new(
  cfg : @ssacfg.SsaCfg,
  allocation : @hashmap.T[Var, RegRef],
  name_var : Var
) -> CodegenBlock {
  let is_entrance = name_var.id == 0
  let name = generate_name(name_var)
  // 1. generate prologue
  let cur_fn : AssemblyFunction = { name, export: is_entrance, body: [] }
  cur_fn.body.append(
    [
      Label(name),
      Comment("args: " + cfg.fn_args[name_var].unwrap().to_string()),
    ],
  )
  if is_entrance {
    cur_fn.body.append([Comment("set up stack"), La(Sp, "stack_space_end")])
  }

  // 2. collect dirtied regs
  let dirtied = @hashset.new()
  let params = cfg.fn_args[name_var].unwrap()
  for param in params {
    let alloc_reg = allocation[param].unwrap()
    if cfg.spilled.contains(param) {
      @util.die("param \{param} of fn \{name_var} spilled")
    } else {
      dirtied.insert(alloc_reg)
    }
  }

  // 3. collect spilled vars
  let mut stack_offset = 0
  let spilled_offset : @hashmap.T[Var, Int] = @hashmap.new()
  for spilled_var in cfg.spilled[name_var].unwrap() {
    // NOTE: stack may be unaligned.
    stack_offset -= get_reg_ty(Var(spilled_var)).byte_size()
    spilled_offset[spilled_var] = stack_offset
  }
  {
    cfg,
    allocation,
    cur_fn: { val: cur_fn },
    block_label: name_var,
    stack_offset,
    dirtied,
    spilled_offset,
  }
}

fn CodegenBlock::find_reg(self : CodegenBlock, var : Var) -> RegRef {
  match self.allocation[var] {
    None => I(Zero) // a var assigned but never used
    Some(reg) => reg
  }
}

fn CodegenBlock::insert_asm(self : CodegenBlock, asm : RvAsm) -> Unit {
  self.cur_fn.val.body.push(asm)
  match asm {
    Add(reg, _, _)
    | Sub(reg, _, _)
    | Xor(reg, _, _)
    | Or(reg, _, _)
    | And(reg, _, _)
    | Sll(reg, _, _)
    | Srl(reg, _, _)
    | Sra(reg, _, _)
    | Slt(reg, _, _)
    | Sltu(reg, _, _)
    | Addi(reg, _, _)
    | Xori(reg, _, _)
    | Ori(reg, _, _)
    | Andi(reg, _, _)
    | Slli(reg, _, _)
    | Srli(reg, _, _)
    | Srai(reg, _, _)
    | Slti(reg, _, _)
    | Sltiu(reg, _, _)
    | Lb(reg, _)
    | Lh(reg, _)
    | Lw(reg, _)
    | Ld(reg, _)
    | Lbu(reg, _)
    | Lhu(reg, _)
    | Lwu(reg, _)
    | Mul(reg, _, _)
    | Mulw(reg, _, _)
    | Mulh(reg, _, _)
    | Mulhsu(reg, _, _)
    | Mulhu(reg, _, _)
    | Div(reg, _, _)
    | Divw(reg, _, _)
    | Divu(reg, _, _)
    | Rem(reg, _, _)
    | Remw(reg, _, _)
    | Remu(reg, _, _)
    | FeqD(reg, _, _)
    | FleD(reg, _, _)
    | Seqz(reg, _)
    | FmvXD(reg, _)
    | La(reg, _) | Li(reg, _) | Neg(reg, _) | Mv(reg, _) | Fcvtwd(reg, _) =>
      self.dirtied.insert(I(reg))
    FaddD(freg, _, _)
    | FsubD(freg, _, _)
    | FmulD(freg, _, _)
    | FdivD(freg, _, _)
    | Fld(freg, _)
    | FmvDX(freg, _)
    | FnegD(freg, _) | FmvD(freg, _) | Fcvtdw(freg, _) | Fsgnjxd(freg, _, _) =>
      self.dirtied.insert(F(freg))
    _ => ()
  }
}

fn CodegenBlock::insert_asms(self : CodegenBlock, asms : Array[RvAsm]) -> Unit {
  for asm in asms {
    self.insert_asm(asm)
  }
}

fn CodegenBlock::assign_i(
  self : CodegenBlock,
  var : Var,
  todo : (Reg) -> Array[RvAsm]
) -> Unit {
  if get_reg_ty(Var(var)) == Void {
    return
  }
  guard let I(target) = self.find_reg(var) else {
    reg => @util.die("assigning to \{reg} with `assign_i` @ \{var}")
  }
  self.insert_asms(todo(target))
}

fn CodegenBlock::assign_f(
  self : CodegenBlock,
  var : Var,
  todo : (FReg) -> Array[RvAsm]
) -> Unit {
  guard let F(target) = self.find_reg(var) else {
    I(Zero) => return  // meaning the value is not used
    reg => @util.die("assigning to \{reg} with `assign_f` @ \{var}")
  }
  self.insert_asms(todo(target))
}

fn CodegenBlock::pull_val_f(self : CodegenBlock, val : Value) -> FReg {
  match val {
    Var(var) => {
      guard let F(reg) = self.allocation[var].unwrap() else {
        _ => @util.die("trying to pull \{val} on to f reg")
      }
      reg
    }
    Double(f) => {
      self.insert_asms(
        [
          Comment("IEEE754 representation of \{f}"),
          Li(reg_swap, f.reinterpret_as_u64().to_string()),
          FmvDX(freg_swap, reg_swap),
        ],
      )
      freg_swap
    }
    _ => @util.die("trying to pull \{val} on to f reg")
  }
}

fn CodegenBlock::pull_val_i(self : CodegenBlock, val : Value) -> Reg {
  match val {
    Var(var) => {
      if var.ty.val == Unit {
        return Zero
      }
      guard let I(reg) = self.allocation[var].unwrap() else {
        _ => @util.die("trying to pull \{val} on to i reg")
      }
      reg
    }
    Label(lbl) => {
      self.insert_asm(La(reg_swap, lbl.to_string()))
      reg_swap
    }
    Unit | Int(0) | Bool(false) => Zero
    Int(i) => {
      self.insert_asm(Li(reg_swap, i.to_string()))
      reg_swap
    }
    Bool(true) => {
      self.insert_asm(Li(reg_swap, "1"))
      reg_swap
    }
    Double(f) => {
      self.insert_asms(
        [
          Comment("IEEE754 representation of \{f}"),
          Li(reg_swap, f.reinterpret_as_u64().to_string()),
        ],
      )
      reg_swap
    }
  }
}

fn CodegenBlock::store_val(
  self : CodegenBlock,
  val : Value,
  address : MemAccess[Reg, Int]
) -> Unit {
  match get_reg_ty(val) {
    F64 => {
      let freg = self.pull_val_f(val)
      self.insert_asm(Fsd(freg, address))
    }
    I32 => {
      let reg = self.pull_val_i(val)
      self.insert_asm(Sw(reg, address))
    }
    PTR64 => {
      let reg = self.pull_val_i(val)
      self.insert_asm(Sd(reg, address))
    }
    Void => ()
  }
}

fn clone_map[A : Hash + Eq, B](m : @hashmap.T[A, B]) -> @hashmap.T[A, B] {
  m.iter() |> @hashmap.from_iter()
}

fn clone_set[A : Hash + Eq](s : @hashset.T[A]) -> @hashset.T[A] {
  s.iter() |> @hashset.from_iter()
}

fn CodegenBlock::branch_to(self : CodegenBlock, target : Var) -> CodegenBlock {
  let branched = {
    ..self,
    block_label: target,
    dirtied: clone_set(self.dirtied),
    spilled_offset: clone_map(self.spilled_offset),
  }
  branched.insert_asm(Label(target.to_string()))
  branched
}

fn CodegenBlock::mov_reg(
  self : CodegenBlock,
  target : RegRef,
  source : RegRef
) -> Unit {
  self.insert_asm(
    match (target, source) {
      (I(target), I(source)) => Mv(target, source)
      (I(target), F(source)) => FmvXD(target, source)
      (F(target), I(source)) => FmvDX(target, source)
      (F(target), F(source)) => FmvD(target, source)
    },
  )
}

// NOTE: this is not optimal but I assume good enough
fn CodegenBlock::resolve_loop(
  self : CodegenBlock,
  args_regs_list : Array[(Value, RegRef)]
) -> Unit {
  let should_replace : @hashmap.T[RegRef, RegRef] = @hashmap.new()
  let should_be_replaced : @hashmap.T[RegRef, RegRef] = @hashmap.new()
  let in_place : @hashset.T[RegRef] = @hashset.new()
  for item in args_regs_list {
    let (val, reg) = item
    // NOTE: we may overwrite tmp regs, so we do this after all regs are pulled 
    guard let Var(_) = val else { _ => continue }
    let pulled_reg = match val.get_type() {
      Double => F(self.pull_val_f(val))
      _ => I(self.pull_val_i(val))
    }
    if pulled_reg != reg {
      should_replace[pulled_reg] = reg
      should_be_replaced[reg] = pulled_reg
    }
  }
  for item in args_regs_list {
    let (val, reg_detect) = item
    if in_place.contains(reg_detect) {
      continue
    }
    let mut chain_end = reg_detect
    let mut is_cycle = false
    while should_replace[chain_end] != None {
      chain_end = should_replace[chain_end].unwrap()
      if chain_end == reg_detect {
        is_cycle = true
        break
      }
    }
    fn fetch_all_regs(start : RegRef) {
      let ret = []
      ret.push(start)
      loop should_be_replaced[start] {
        None => break ret
        Some(reg) =>
          if reg == start {
            break ret
          } else {
            ret.push(reg)
            continue should_be_replaced[reg]
          }
      }
    }

    let all_regs = fetch_all_regs(chain_end)
    for reg_moved in all_regs {
      in_place.insert(reg_moved)
    }
    if is_cycle {
      self.cur_fn.val.body.push(Comment("cycle: \{all_regs}"))
      self.mov_reg(I(reg_swap), chain_end)
      loop (should_be_replaced[chain_end], chain_end) {
        (None, _) => @util.die("not a loop")
        (Some(replacer), replaced) => {
          if replacer == reg_detect {
            self.mov_reg(replaced, I(reg_swap))
            break
          }
          self.mov_reg(replaced, replacer)
          continue (should_be_replaced[replacer], replacer)
        }
      }
    } else {
      self.cur_fn.val.body.push(Comment("chain: \{all_regs}"))
      loop (should_be_replaced[chain_end], chain_end) {
        (None, _) => break
        (Some(replacer), replaced) => {
          self.mov_reg(replaced, replacer)
          continue (should_be_replaced[replacer], replacer)
        }
      }
    }
    // writing to cur reg still may affect other regs that needs value on us, so 
    // we do this after chain is done
    guard let Var(_) = val else {
      _ => {
        self.cur_fn.val.body.push(
          Comment("pull immediate \{val} on to \{reg_detect}"),
        )
        let pulled_reg = match val.get_type() {
          Double => F(self.pull_val_f(val))
          _ => I(self.pull_val_i(val))
        }
        self.mov_reg(reg_detect, pulled_reg)
        continue
      }
    }

  }
}

fn get_tuple_offset(t : T, idx : Int) -> Int {
  // we may pass in a function
  // then it's converted to tuple during closure conversion
  match t {
    Tuple(tys) => {
      let mut result = 0
      for i = 0; i < idx; i = i + 1 {
        result += ty_get_reg_ty(tys[i]).byte_size()
      }
      return result
    }
    Fun(_) => if idx == 0 { return 0 } else { return PTR64.byte_size() }
    _ => @util.die("unpacking a non tuple \{t} @ \{idx}")
  }
}

fn CodegenBlock::new_label(self : CodegenBlock, prefix : String) -> String {
  let var = self.cfg.new_named(prefix)
  var.to_string()
}

// PERF: proritize the use of saved regs in stead of putting onto stack
// take care not to introduce a chain reaction where you dirty s regs

// BUG: we didn't consider spilled registers here
fn CodegenBlock::call_c_conv_aligned(
  self : CodegenBlock,
  target : String,
  args : Array[Value],
  reg_ret : RegRef
) -> Unit {
  // 1. Store all modified regs
  let mut tmp_stack_offset = self.stack_offset
  let to_store = self.dirtied
    .iter()
    .filter(
      fn(reg) {
        not(
          is_reg_saved(reg) || [I(reg_swap), F(freg_swap), I(Sp)].contains(reg),
        )
      },
    )
    .collect()
  for reg in to_store {
    match reg {
      I(reg) => { // NOTE: we have no idea if it's a pointer or int on that reg
        tmp_stack_offset -= PTR64.byte_size()
        self.cur_fn.val.body.push(
          Sd(reg, { base: Sp, offset: tmp_stack_offset }),
        )
      }
      F(freg) => {
        tmp_stack_offset -= F64.byte_size()
        self.cur_fn.val.body.push(
          Fsd(freg, { base: Sp, offset: tmp_stack_offset }),
        )
      }
    }
  }
  // 2. Fill in the params to regs
  let args_regs_list = resolve_arg_regs(args)
  // 2. Generate an assign order
  //  WARN: these steps may dirty too many regs
  //  so we don't use insert_asm/insert_asms
  self.resolve_loop(args_regs_list)
  // 3. Generate the call
  let stack_offset_aligned = (tmp_stack_offset - 15) / 16 * 16
  if stack_offset_aligned == 0 {
    self.cur_fn.val.body.push(Call(target))
  } else {
    self.cur_fn.val.body.append(
      [
        Addi(Sp, Sp, stack_offset_aligned),
        Call(target),
        Addi(Sp, Sp, -stack_offset_aligned),
      ],
    )
  }

  // 4. move the return value to somewhere else, this has to be done before loading the regs
  // this step does dirty.
  if reg_ret != I(A0) {
    self.mov_reg(reg_ret, I(A0))
  }

  // 5. load all modifed regs
  to_store.rev_each(
    fn(stored) {
      if stored == reg_ret {
        return  // if we have a that reg occupied for return value, skip it.
      }
      //self.dirtied.remove(stored)
      match stored {
        I(reg) => { // NOTE: we have no idea if it's a pointer or int on that reg
          self.cur_fn.val.body.push(
            Ld(reg, { base: Sp, offset: tmp_stack_offset }),
          )
          tmp_stack_offset += PTR64.byte_size()
        }
        F(freg) => {
          self.cur_fn.val.body.push(
            Fld(freg, { base: Sp, offset: tmp_stack_offset }),
          )
          tmp_stack_offset += F64.byte_size()
        }
      }
    },
  )
}

fn CodegenBlock::codegen(self : CodegenBlock) -> Unit {
  let block = self.cfg.blocks[self.block_label].unwrap()
  // as long as the entrance of the body is the first block, our codegen order doesn't matter too much.
  // 1. Do all mallocs for tuples together
  let malloc_offsets : @hashmap.T[Var, Int] = @hashmap.new()
  let mut malloc_offset_cur = 0
  for inst in block.insts {
    guard let MakeTuple(bind, vals) = inst else {
      _ => continue
      // 1. calculuate tuple size
    }
    let mut tuple_byte_size = 0
    for val in vals {
      tuple_byte_size += get_reg_ty(val).byte_size()
    }
    malloc_offsets[bind] = malloc_offset_cur
    malloc_offset_cur += tuple_byte_size
  }
  if malloc_offset_cur > 0 {
    // Just allocate on stack, because we don't use them as call stack
    self.insert_asm(Addi(Sp, Sp, -malloc_offset_cur))
  }

  // 2. generate asms for each inst
  for inst in block.insts {
    self.insert_asm(Comment(inst.to_string()))
    match inst {
      Copy(bind, copied) =>
        match get_reg_ty(copied) {
          F64 => {
            let freg_copied = self.pull_val_f(copied)
            self.assign_f(bind, fn(freg) { [FmvD(freg, freg_copied)] })
          }
          _ => {
            let reg_copied = self.pull_val_i(copied)
            self.assign_i(bind, fn(reg) { [Mv(reg, reg_copied)] })
          }
        }
      Load(var) => {
        let offset = self.spilled_offset[var].unwrap()
        match get_reg_ty(Var(var)) {
          F64 =>
            self.assign_f(var, fn(freg) { [Fld(freg, { base: Sp, offset })] })
          I32 => self.assign_i(var, fn(reg) { [Lw(reg, { base: Sp, offset })] })
          PTR64 =>
            self.assign_i(var, fn(reg) { [Ld(reg, { base: Sp, offset })] })
          Void => self.assign_i(var, fn(reg) { [Mv(reg, Zero)] })
        }
      }
      Store(var) => {
        let offset = self.spilled_offset[var].unwrap()
        self.store_val(Var(var), { base: Sp, offset })
      }
      MakeTuple(bind, vals) => {
        let mut reg_tup = Zero
        // 1. decide the malloced pointer's address
        self.assign_i(
          bind,
          fn(reg) {
            reg_tup = reg
            [Addi(reg, Sp, malloc_offsets[bind].unwrap())]
          },
        )
        // 2. store values into tuple
        let mut offset = 0
        for val in vals {
          self.store_val(val, { base: reg_tup, offset })
          offset += get_reg_ty(val).byte_size()
        }
      }
      KthTuple(bind, tup, offset) => {
        let tup_reg = self.pull_val_i(tup)
        match get_reg_ty(Var(bind)) {
          F64 =>
            self.assign_f(
              bind,
              fn(freg) {
                [
                  Fld(
                    freg,
                    {
                      base: tup_reg,
                      offset: get_tuple_offset(tup.get_type(), offset),
                    },
                  ),
                ]
              },
            )
          I32 | Void =>
            self.assign_i(
              bind,
              fn(reg) {
                [
                  Lw(
                    reg,
                    {
                      base: tup_reg,
                      offset: get_tuple_offset(tup.get_type(), offset),
                    },
                  ),
                ]
              },
            )
          PTR64 =>
            self.assign_i(
              bind,
              fn(reg) {
                [
                  Ld(
                    reg,
                    {
                      base: tup_reg,
                      offset: get_tuple_offset(tup.get_type(), offset),
                    },
                  ),
                ]
              },
            )
        }
      }
      Prim(bind, Not, [v]) => {
        let reg_v = self.pull_val_i(v)
        self.assign_i(bind, fn { reg_bind => [Xori(reg_bind, reg_v, 1)] })
      }
      Prim(bind, Neg(Double), [v]) => {
        let reg_v = self.pull_val_f(v)
        self.assign_f(bind, fn { reg_bind => [FnegD(reg_bind, reg_v)] })
      }
      Prim(bind, Neg(Int), [v]) => {
        let reg_v = self.pull_val_i(v)
        self.assign_i(bind, fn { reg_bind => [Neg(reg_bind, reg_v)] })
      }
      Prim(bind, IntOfFloat, [f]) => {
        let reg_v = self.pull_val_f(f)
        self.assign_i(bind, fn { reg_bind => [Fcvtwd(reg_bind, reg_v)] })
      }
      Prim(bind, FloatOfInt, [i]) => {
        let reg_v = self.pull_val_i(i)
        self.assign_f(bind, fn { reg_bind => [Fcvtdw(reg_bind, reg_v)] })
      }
      Prim(bind, AbsFloat, [f]) => {
        let reg_v = self.pull_val_f(f)
        self.assign_f(
          bind,
          fn { reg_bind => [Fsgnjxd(reg_bind, reg_v, reg_v)] },
        )
      }
      // TODO: may generate more higher quality asm if idx is known at compile time
      Prim(bind, Get, [arr, idx]) => {
        let reg_idx = self.pull_val_i(idx)
        let reg_arr = self.pull_val_i(arr)
        match get_reg_ty(Var(bind)) {
          Void => self.assign_i(bind, fn { reg => [Mv(reg, Zero)] })
          F64 =>
            self.assign_f(
              bind,
              fn {
                reg_bind =>
                  [
                    // HACK: arr may not be swap reg but idx may be.
                    // this is due to we have no array literals
                    Slli(reg_swap, reg_idx, F64.byte_size_bits()),
                    Add(reg_swap, reg_swap, reg_arr),
                    Fld(reg_bind, { base: reg_swap, offset: 0 }),
                  ]
              },
            )
          I32 =>
            self.assign_i(
              bind,
              fn {
                reg_bind =>
                  [
                    // HACK: arr may not be swap reg but idx may be.
                    // this is due to we have no array literals
                    Slli(reg_swap, reg_idx, I32.byte_size_bits()),
                    Add(reg_swap, reg_swap, reg_arr),
                    Lw(reg_bind, { base: reg_swap, offset: 0 }),
                  ]
              },
            )
          PTR64 =>
            self.assign_i(
              bind,
              fn {
                reg_bind =>
                  [
                    // HACK: arr may not be swap reg but idx may be.
                    // this is due to we have no array literals
                    Slli(reg_swap, reg_idx, PTR64.byte_size_bits()),
                    Add(reg_swap, reg_swap, reg_arr),
                    Ld(reg_bind, { base: reg_swap, offset: 0 }),
                  ]
              },
            )
        }
      }
      Prim(bind, GetArrPtr, [arr, idx]) => {
        let reg_idx = self.pull_val_i(idx)
        let reg_arr = self.pull_val_i(arr)
        guard let Array(elem_ty) = arr.get_type()
        let elem_ty = ty_get_reg_ty(elem_ty)
        self.assign_i(
          bind,
          fn(reg) {
            [
              Slli(reg_swap, reg_idx, elem_ty.byte_size_bits()),
              Add(reg, reg_swap, reg_arr),
            ]
          },
        )
      }
      Prim(_, StoreToPtr, [rhs, ptr]) => {
        let reg_ptr = self.pull_val_i(ptr)
        self.store_val(rhs, { base: reg_ptr, offset: 0 })
      }
      Prim(bind, Math(op, Int), [lhs, rhs]) =>
        self.assign_i(
          bind,
          fn(reg) {
            let reg_lhs = self.pull_val_i(lhs)
            // NOTE: whenever we pull 2 regs and both of them maybe tmp, 
            // we replace the second tmp with target
            let reg_rhs = self.pull_val_i(rhs)
            match op {
              Add => [Add(reg, reg_lhs, reg_rhs)]
              Sub => [Sub(reg, reg_lhs, reg_rhs)]
              Mul => [Mulw(reg, reg_lhs, reg_rhs)]
              Div => [Divw(reg, reg_lhs, reg_rhs)]
            }
          },
        )
      Prim(bind, Math(op, Double), [lhs, rhs]) =>
        self.assign_f(
          bind,
          fn(reg) {
            let reg_lhs = self.pull_val_f(lhs)
            let reg_rhs = self.pull_val_f(rhs)
            match op {
              Add => [FaddD(reg, reg_lhs, reg_rhs)]
              Sub => [FsubD(reg, reg_lhs, reg_rhs)]
              Mul => [FmulD(reg, reg_lhs, reg_rhs)]
              Div => [FdivD(reg, reg_lhs, reg_rhs)]
            }
          },
        )
      Prim(bind, Eq, [lhs, rhs]) =>
        match lhs.get_type() {
          Double => {
            let reg_lhs = self.pull_val_f(lhs)
            let reg_rhs = self.pull_val_f(rhs)
            // NOTE: since we've reserve freg first, this won't happen
            if reg_lhs == reg_rhs {
              @util.die("pulling same reg for comparing floats")
            }
            self.assign_i(bind, fn(reg) { [FeqD(reg, reg_lhs, reg_rhs)] })
          }
          _ => {
            let reg_lhs = self.pull_val_i(lhs)
            self.assign_i(
              bind,
              fn(reg) {
                let reg_rhs = self.pull_val_i(rhs)
                [Xor(reg_swap, reg_lhs, reg_rhs), Seqz(reg, reg_swap)]
              },
            )
          }
        }
      Prim(bind, Le, [lhs, rhs]) =>
        match lhs.get_type() {
          Double => {
            let reg_lhs = self.pull_val_f(lhs)
            let reg_rhs = self.pull_val_f(rhs)
            // NOTE: since we've reserve freg first, this won't happen
            if reg_lhs == reg_rhs {
              @util.die("pulling same reg for comparing floats")
            }
            self.assign_i(bind, fn(reg) { [FleD(reg, reg_lhs, reg_rhs)] })
          }
          _ => {
            let reg_lhs = self.pull_val_i(lhs)
            self.assign_i(
              bind,
              // HACK: we're using 32bits int so this won't overflow
              fn(reg) {
                let reg_rhs = self.pull_val_i(rhs)
                [
                  Sub(reg, reg_lhs, reg_rhs),
                  Addi(reg, reg, -1),
                  Srli(reg, reg, 63), // take sign bit
                ]
              },
            )
          }
        }
      Prim(_) => @util.die("malformed prim call \{inst}")
    }
  }
  self.insert_asm(Comment(block.last_inst.val.to_string()))
  match block.last_inst.val {
    Branch(cond, blk_then, blk_else) => {
      let reg_cond = self.pull_val_i(cond)
      self.insert_asms([Beq(Zero, reg_cond, blk_else.to_string())])
      self.branch_to(blk_then).codegen()
      self.insert_asm(
        Comment(
          "we have tail call so no point generating a jump to skip the else branch",
        ),
      )
      self.branch_to(blk_else).codegen()
    }
    BranchEq(lhs, rhs, blk_then, blk_else) =>
      match get_reg_ty(lhs) {
        // TODO: just like Eq & Le, we need to ensure the 2 regs doesn't crash
        F64 => {
          let reg_lhs = self.pull_val_f(lhs)
          let reg_rhs = self.pull_val_f(rhs)
          // NOTE: since we've reserve freg first, this won't happen
          if reg_lhs == reg_rhs {
            @util.die("pulling same reg for comparing floats")
          }
          self.insert_asms(
            [
              FeqD(reg_swap, reg_lhs, reg_rhs),
              Beq(reg_swap, Zero, blk_else.to_string()),
            ],
          )
          self.branch_to(blk_then).codegen()
          self.insert_asm(
            Comment(
              "we have tail call so no point generating a jump to skip the else branch",
            ),
          )
          self.branch_to(blk_else).codegen()
        }
        _ => {
          let reg_lhs = self.pull_val_i(lhs)
          let reg_rhs = self.pull_val_i(rhs)
          // NOTE: since we've reserve freg first, this won't happen
          if reg_lhs == reg_rhs {
            @util.die("pulling same reg for comparing floats")
          }
          self.insert_asm(Bne(reg_lhs, reg_rhs, blk_else.to_string()))
          self.branch_to(blk_then).codegen()
          self.insert_asm(
            Comment(
              "we have tail call so no point generating a jump to skip the else branch",
            ),
          )
          self.branch_to(blk_else).codegen()
        }
      }
    BranchLe(lhs, rhs, blk_then, blk_else) =>
      match get_reg_ty(lhs) {
        // TODO: just like Eq & Le, we need to ensure the 2 regs doesn't crash
        F64 => {
          let reg_lhs = self.pull_val_f(lhs)
          let reg_rhs = self.pull_val_f(rhs)
          // NOTE: since we've reserve freg first, this won't happen
          if reg_lhs == reg_rhs {
            @util.die("pulling same reg for comparing floats")
          }
          self.insert_asms(
            [
              FleD(reg_swap, reg_lhs, reg_rhs),
              Beq(reg_swap, Zero, blk_else.to_string()),
            ],
          )
          self.branch_to(blk_then).codegen()
          self.insert_asm(
            Comment(
              "we have tail call so no point generating a jump to skip the else branch",
            ),
          )
          self.branch_to(blk_else).codegen()
        }
        _ => {
          let reg_lhs = self.pull_val_i(lhs)
          let reg_rhs = self.pull_val_i(rhs)
          // NOTE: since we've reserve freg first, this won't happen
          if reg_lhs == reg_rhs {
            @util.die("pulling same reg for comparing floats")
          }
          self.insert_asm(Bgt(reg_lhs, reg_rhs, blk_else.to_string()))
          self.branch_to(blk_then).codegen()
          self.insert_asm(
            Comment(
              "we have tail call so no point generating a jump to skip the else branch",
            ),
          )
          self.branch_to(blk_else).codegen()
        }
      }
    Exit => self.insert_asms([Li(A0, "0"), Li(A7, "93"), Ecall])
    MakeArray(len, elem, continuation) => {
      // 1. call to generate an array
      // we keep the result at 1st reg so we call the continuation immediately
      let create_array_fn = "minimbt_" +
        (match get_reg_ty(elem) {
          F64 => "create_float_array"
          I32 | Void => "create_array"
          PTR64 => "create_ptr_array"
        })
      self.call_c_conv_aligned(create_array_fn, [len, elem], I(A0))

      // 2. prepare regs for continuation. A0 is already the array
      self.resolve_loop([(continuation, I(A1))])
      // 3. Generate the call to continuation
      let cont_reg = self.pull_val_i(continuation)
      self.insert_asms(
        [Ld(cont_reg, { base: cont_reg, offset: 0 }), Jr(cont_reg)],
      )
    }
    Call(f, args) => {
      // 1. Fill in the params to regs
      let args_regs_list = resolve_arg_regs(args)
      // 2. Generate an assign order
      self.resolve_loop(args_regs_list)
      // 3. Generate the call
      match f {
        Label(address) => self.insert_asm(J(address.to_string()))
        Var(_) => {
          let reg_f = self.pull_val_i(f)
          self.insert_asm(Jr(reg_f))
        }
        _ => @util.die("Calling non function \{f}")
      }
    }
  }
}
