enum NumOrInf {
  Fin(Int)
  Inf
}

fn NumOrInf::less_than(self : NumOrInf, rhs : Int) -> Bool {
  match self {
    Inf => false
    Fin(i) => i < rhs
  }
}

fn NumOrInf::dec(self : NumOrInf) -> NumOrInf {
  match self {
    Inf => Inf
    Fin(i) => Fin(i - 1)
  }
}

fn interference_graph_simplify[Color](
  all_vars : @hashset.T[Var],
  interferences : InterferenceGraph,
  colors : Array[Color],
  pre_coloring : @hashmap.T[Var, Color],
  simplify_stack : Array[Var]
) -> Bool { // return whether or not we need to spill
  let working_queue : @queue.T[Var] = @queue.new()
  let interfere_cnt : @hashmap.T[Var, NumOrInf] = @hashmap.new()
  let can_allocate : @hashset.T[Var] = @hashset.new()
  for var in all_vars {
    let num_interfered = if pre_coloring.contains(var) {
      Inf
    } else {
      Fin(interferences[var].map(fn(set) { set.size() }).or(0))
    }
    interfere_cnt[var] = num_interfered
    if num_interfered.less_than(colors.length()) {
      working_queue.push(var)
      can_allocate.insert(var)
    }
  }
  loop working_queue.pop() {
    None => break
    Some(var) => {
      simplify_stack.push(var)
      // remove var from graph
      for neighbor in interferences[var].unwrap() {
        let interfere_new = interfere_cnt[neighbor].unwrap().dec()
        // update the graph
        interferences[neighbor].unwrap().remove(var)
        interfere_cnt[neighbor] = interfere_new
        if interfere_new.less_than(colors.length()) &&
          not(can_allocate.contains(neighbor)) {
          working_queue.push(neighbor)
          can_allocate.insert(neighbor)
        }
      }
      continue working_queue.pop()
    }
  }
  return can_allocate.size() + pre_coloring.size() < all_vars.size()
}
