typealias InterferenceGraph = @hashmap.T[Var, @hashset.T[Var]]

fn collect_relations(g : InterferenceGraph, var_set : @hashset.T[Var]) -> Unit {
  for var in var_set {
    let cur_set = var_set.iter() |> @hashset.from_iter()
    cur_set.remove(var)
    match g[var] {
      None => g[var] = cur_set
      Some(orig_set) => g[var] = orig_set.union(cur_set)
    }
  }
}

struct LiveVarAnalysis {
  var_set : @hashset.T[Var]
  fvar_set : @hashset.T[Var]
  all_var_i : @hashset.T[Var]
  all_var_f : @hashset.T[Var]
}

fn LiveVarAnalysis::new() -> LiveVarAnalysis {
  {
    var_set: @hashset.new(),
    fvar_set: @hashset.new(),
    all_var_i: @hashset.new(),
    all_var_f: @hashset.new(),
  }
}

fn LiveVarAnalysis::clone(self : LiveVarAnalysis) -> LiveVarAnalysis {
  {
    ..self,
    var_set: self.var_set.iter() |> @hashset.from_iter(),
    fvar_set: self.fvar_set.iter() |> @hashset.from_iter(),
  }
}

fn LiveVarAnalysis::union(
  self : LiveVarAnalysis,
  other : LiveVarAnalysis
) -> LiveVarAnalysis {
  {
    var_set: self.var_set.union(other.var_set),
    fvar_set: self.fvar_set.union(other.fvar_set),
    all_var_i: self.all_var_i.union(other.all_var_i),
    all_var_f: self.all_var_f.union(other.all_var_f),
  }
}

fn LiveVarAnalysis::fill_interference(
  self : LiveVarAnalysis,
  g_i : InterferenceGraph,
  g_f : InterferenceGraph
) -> Unit {
  collect_relations(g_i, self.var_set)
  collect_relations(g_f, self.fvar_set)
}

fn LiveVarAnalysis::collect_val(self : LiveVarAnalysis, v : Value) -> Unit {
  match v {
    Var(v) =>
      match v.ty.val {
        Double => {
          self.fvar_set.insert(v)
          self.all_var_f.insert(v)
        }
        _ => {
          self.var_set.insert(v)
          self.all_var_i.insert(v)
        }
      }
    _ => ()
  }
}

fn LiveVarAnalysis::collect_pc_inst(
  self : LiveVarAnalysis,
  inst : @ssacfg.PCInst
) -> Unit {
  match inst {
    Branch(cond, _then, _else) => self.collect_val(cond)
    BranchEq(lhs, rhs, _then, _else) | BranchLe(lhs, rhs, _then, _else) => {
      self.collect_val(lhs)
      self.collect_val(rhs)
    }
    Call(f, args) => {
      self.collect_val(f)
      args.each(fn(arg) { self.collect_val(arg) })
    }
    MakeArray(len, elem, kont) => {
      self.collect_val(len)
      self.collect_val(elem)
      self.collect_val(kont)
    }
    Exit => ()
  }
}

fn LiveVarAnalysis::collect_inst(
  self : LiveVarAnalysis,
  inst : @ssacfg.Inst
) -> Unit {
  match inst {
    MakeTuple(bind, tup) => {
      self.var_set.remove(bind)
      tup.each(fn(v) { self.collect_val(v) })
    }
    KthTuple(bind, tup, _) => {
      self.var_set.remove(bind)
      self.collect_val(tup)
    }
    Copy(bind, copied) => {
      self.var_set.remove(bind)
      self.collect_val(copied)
    }
    Prim(bind, _, args) => {
      self.var_set.remove(bind)
      args.each(fn(v) { self.collect_val(v) })
    }
    Load(bind) => self.var_set.remove(bind)
    Store(bind) => self.var_set.insert(bind)
  }
}

fn collect_blocks_of_fn(cfg : @ssacfg.SsaCfg, fn_name : Var) -> Array[Var] {
  let stack_process = []
  let q_collect : @queue.T[Var] = @queue.of([fn_name])
  loop q_collect.pop() {
    None => break stack_process
    Some(label) => {
      let cur_blk = cfg.blocks[label].unwrap()
      stack_process.push(label)
      match cur_blk.last_inst.val {
        Branch(_, blk_then, blk_else)
        | BranchEq(_, _, blk_then, blk_else)
        | BranchLe(_, _, blk_then, blk_else) => {
          q_collect.push(blk_then)
          q_collect.push(blk_else)
        }
        _ => ()
      }
      continue q_collect.pop()
    }
  }
}

fn build_interference_graph(
  cfg : @ssacfg.SsaCfg,
  fn_name : Var
) -> (InterferenceGraph, InterferenceGraph, @hashset.T[Var], @hashset.T[Var]) {
  // returns: 
  // 1. graph for i regs
  // 2. graph for f regs
  // 3. vars used for i regs
  // 4. vars used for f regs

  // process blocks in reverse order(pop)
  let stack_process : Array[Var] = collect_blocks_of_fn(cfg, fn_name)

  // a map from block label to (live_vars_set_i, live_vars_set_f) tuple
  let live_vars_init : @hashmap.T[Var, LiveVarAnalysis] = @hashmap.new()
  let color_graph_i : InterferenceGraph = @hashmap.new()
  let color_graph_f : InterferenceGraph = @hashmap.new()
  let mut all_vars_i_ret = @hashset.new()
  let mut all_vars_f_ret = @hashset.new()
  loop stack_process.pop() {
    Some(lbl) => {
      let blk = cfg.blocks[lbl].unwrap()
      let l = live_vars_init[lbl].or(LiveVarAnalysis::new())
      // collect live vars on last instruction
      l.collect_pc_inst(blk.last_inst.val)
      l.fill_interference(color_graph_i, color_graph_f)
      // collect live vars on all other instructions
      blk.insts.rev_each(
        fn(inst) {
          l.collect_inst(inst)
          l.fill_interference(color_graph_i, color_graph_f)
          // NOTE: deal with special case: tuple target should not use a 
          // register same as any of its elements
          match inst {
            MakeTuple(target, vals) => {
              let l_tmp = LiveVarAnalysis::new()
              vals.each(fn(val) { l_tmp.collect_val(val) })
              l_tmp.var_set.insert(target)
              l_tmp.fill_interference(color_graph_i, color_graph_f)
            }
            _ => ()
          }
        },
      )
      // topo sort algorithm continued
      match blk.from {
        Some(previous) =>
          // accumlate live var set on prebranched blocks
          match live_vars_init[previous] {
            None => live_vars_init[previous] = l.clone()
            Some(already) => live_vars_init[previous] = already.union(l)
          }
        None => ()
      }
      if lbl == fn_name {
        all_vars_i_ret = l.all_var_i
        all_vars_f_ret = l.all_var_f
      }
      let a = stack_process.pop()
      continue a
    }
    None =>
      return (color_graph_i, color_graph_f, all_vars_i_ret, all_vars_f_ret)
  }
}
