// REF: 
// - Modern Compiler Implementation in ML - Andrew W. Appel

fn reg_allocate_on_fn(
  input : @ssacfg.SsaCfg,
  fn_label : Var
) -> @hashmap.T[Var, RegRef] {
  //println("allocation for \{fn_label}")

  // generate pre allocation
  let allocation_i : @hashmap.T[Var, Reg] = @hashmap.new()
  let allocation_f : @hashmap.T[Var, FReg] = @hashmap.new()
  // allocate for args
  let params = input.fn_args[fn_label].unwrap()
  loop (params[:], reg_arg_list[:], freg_arg_list[:]) {
    ([], _, _) => break
    ([param, .. as param_rest], iregs, fregs) =>
      match param.ty.val {
        Double => {
          guard let [freg, .. as fregs] = fregs else {
            _ => @util.die("no i64 register can be assigned to \{param}")
          }
          allocation_f[param] = freg
          continue (param_rest, iregs, fregs)
        }
        _ => {
          guard let [ireg, .. as iregs] = iregs else {
            _ => @util.die("no i64 register can be assigned to \{param}")
          }
          allocation_i[param] = ireg
          continue (param_rest, iregs, fregs)
        }
      }
  }
  let simplify_stack_i = []
  let simplify_stack_f = []
  let spilled_set = match input.spilled[fn_label] {
    None => {
      let out = @hashset.new()
      input.spilled[fn_label] = out
      out
    }
    Some(set) => set
  }
  while true {
    let (graph_i, graph_f, vars_i, vars_f) = build_interference_graph(
      input, fn_label,
    )
    let should_spill_i = interference_graph_simplify(
      vars_i, graph_i, alloc_regs, allocation_i, simplify_stack_i,
    )
    let should_spill_f = interference_graph_simplify(
      vars_f, graph_f, alloc_fregs, allocation_f, simplify_stack_f,
    )
    if should_spill_i {
      let i_spilled = reg_spill_fn(
        input, vars_i, simplify_stack_i, graph_i, fn_label,
      )
      spilled_set.insert(i_spilled)
    }
    if should_spill_f {
      let f_spilled = reg_spill_fn(
        input, vars_f, simplify_stack_f, graph_f, fn_label,
      )
      spilled_set.insert(f_spilled)
    }
    if should_spill_f || should_spill_i {
      continue
    } else {
      break
    }
  }
  let (graph_i, graph_f, _, _) = build_interference_graph(input, fn_label)
  simplify_stack_i.rev_each(
    fn(var) {
      let neighbor_colors = graph_i[var]
        .unwrap()
        .iter()
        .fold(
          init=@hashset.new(),
          fn(acc, var) {
            match allocation_i[var] {
              None => ()
              Some(color) => acc.insert(color)
            }
            acc
          },
        )
      let color = alloc_regs
        .iter()
        .filter(fn(c) { not(neighbor_colors.contains(c)) })
        .head()
      match color {
        None => @util.die("unreachable: can't assign color to \{var}")
        Some(color) => allocation_i[var] = color
      }
    },
  )
  simplify_stack_f.rev_each(
    fn(var) {
      let neighbor_colors = graph_f[var]
        .unwrap()
        .iter()
        .fold(
          init=@hashset.new(),
          fn(acc, var) {
            match allocation_f[var] {
              None => ()
              Some(color) => acc.insert(color)
            }
            acc
          },
        )
      let color = alloc_fregs
        .iter()
        .filter(fn(c) { not(neighbor_colors.contains(c)) })
        .head()
      match color {
        None => @util.die("unreachable: can't assign color to \{var}")
        Some(color) => allocation_f[var] = color
      }
    },
  )
  allocation_i
  .iter()
  .map(fn { (var, reg) => (var, I(reg)) })
  .concat(allocation_f.iter().map(fn { (var, reg) => (var, F(reg)) }))
  |> @hashmap.from_iter()
}
