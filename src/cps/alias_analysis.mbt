fn alias_analysis(c : Cps, optimized : Ref[Bool]) -> Cps {
  fn rec(c : Cps) {
    alias_analysis(c, optimized)
  }

  match c {
    Tuple(arr, bind, inner) => Tuple(arr, bind, rec(inner))
    KthTuple(idx, v, bind, inner) => KthTuple(idx, v, bind, rec(inner))
    // when f1 is an alias for f2, always replace it
    Fix(f1, args1, App(f2, args2), rest) => {
      let args1_fix = args1.map(fn { v => Var(v) })
      if args1_fix == args2 {
        optimized.val = true
        // f1 is an alias of f2
        rec(rest.replace_var_bind(f1, f2))
      } else {
        Fix(f1, args1, App(f2, args2), rec(rest))
      }
    }
    Fix(f, args, body, rest) => Fix(f, args, rec(body), rec(rest))
    If(cond, _then, _else) => If(cond, rec(_then), rec(_else))
    IfEq(lhs, rhs, _then, _else) => IfEq(lhs, rhs, rec(_then), rec(_else))
    IfLe(lhs, rhs, _then, _else) => IfLe(lhs, rhs, rec(_then), rec(_else))
    Prim(op, vs, bind, rest) => Prim(op, vs, bind, rec(rest))
    MakeArray(_) | Exit | App(_) => c
  }
}
