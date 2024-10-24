fn beta_contraction(c : Cps) -> Cps {
  let rec = beta_contraction
  match c {
    Tuple(arr, bind, inner) => Tuple(arr, bind, rec(inner))
    KthTuple(idx, v, bind, inner) => KthTuple(idx, v, bind, rec(inner))
    Fix(f1, args1, App(f2, args2), rest) => {
      let args1_fix = args1.map(fn { v => Var(v) })
      if args1_fix == args2 {
        // f1 is an alias of f2
        rec(rest.replace_var_bind(f1, f2))
      } else {
        Fix(f1, args1, App(f2, args2), rec(rest))
      }
    }
    Fix(f, args, body, rest) => Fix(f, args, rec(body), rec(rest))
    If(cond, _then, _else) => If(cond, rec(_then), rec(_else))
    Prim(op, vs, bind, rest) => Prim(op, vs, bind, rec(rest))
    c => c
  }
}

pub fn optimize_cps(c : Cps) -> Cps {
  c |> beta_contraction()
}
