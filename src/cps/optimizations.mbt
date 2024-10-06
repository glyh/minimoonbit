pub fn alias_analysis(c : Cps) -> Cps {
  let rec = alias_analysis
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
    Switch(v, branches) => Switch(v, branches.map(rec))
    Prim(op, vs, bind, rest) => Prim(op, vs, bind, rec(rest))
    c => c
  }
}

test "array equal" {
  let a1 = [1, 2, 3, 4]
  let a2 = [1, 2, 3]
  a2.push(4)
  assert_eq!(a1, a2)
}
