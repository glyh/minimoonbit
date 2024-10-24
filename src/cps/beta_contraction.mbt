struct BetaContractInfo {
  referred_as_closure : Bool
  num_called : Int
} derive(Show)

fn BetaContractInfo::merge_result(
  self : BetaContractInfo,
  other : BetaContractInfo
) -> BetaContractInfo {
  {
    num_called: self.num_called + other.num_called,
    referred_as_closure: self.referred_as_closure || other.referred_as_closure,
  }
}

fn beta_contract_info(c : Cps, f : Var) -> BetaContractInfo {
  let rec = beta_contract_info
  fn val_is_f(v : Value) {
    v == Var(f) || v == Label(f)
  }

  fn vals_contains_f(vs : Array[Value]) {
    vs.fold(init=false, fn(acc, ele) { acc || val_is_f(ele) })
  }

  match c {
    Tuple(vals, _, rest) | Prim(_, vals, _, rest) =>
      { num_called: 0, referred_as_closure: vals_contains_f(vals) }.merge_result(
        rec(rest, f),
      )
    KthTuple(_, _, _, inner) => rec(inner, f)
    Fix(_, _, c1, c2) | If(_, c1, c2) | IfLe(_, _, c1, c2) | IfEq(_, _, c1, c2) =>
      rec(c1, f).merge_result(rec(c2, f))
    App(g, args) =>
      {
        referred_as_closure: vals_contains_f(args),
        num_called: if val_is_f(g) {
          1
        } else {
          0
        },
      }
    MakeArray(_, elem, cont) =>
      { num_called: 0, referred_as_closure: val_is_f(elem) || val_is_f(cont) }
    Exit => { referred_as_closure: false, num_called: 0 }
  }
}

fn expand_function(
  f : Var,
  params : Array[Var],
  body : Cps,
  target : Cps
) -> Cps {
  let rec = fn(target) { expand_function(f, params, body, target) }
  match target {
    Tuple(tup, bind, rest) => Tuple(tup, bind, rec(rest))
    KthTuple(k, tup, bind, rest) => KthTuple(k, tup, bind, rec(rest))
    Fix(f2, args2, body2, rest2) => Fix(f2, args2, rec(body2), rec(rest2))
    If(cond, _then, _else) => If(cond, rec(_then), rec(_else))
    IfLe(lhs, rhs, _then, _else) => IfLe(lhs, rhs, rec(_then), rec(_else))
    IfEq(lhs, rhs, _then, _else) => IfEq(lhs, rhs, rec(_then), rec(_else))
    Prim(op, args, bind, rest) => Prim(op, args, bind, rec(rest))
    App(f2, args2) =>
      if f2 == Label(f) || f2 == Var(f) {
        // NOTE: when extracting to generate multiple copy, 
        // we need to ensure each copy has distinctive ids
        // I don't have them implemented here, however 
        let a = zip2(params, args2).fold(
          init=body,
          fn(acc, ele) {
            let (arg, replaced) = ele
            acc.replace_var_bind(arg, replaced)
          },
        )
        a
      } else {
        target
      }
    Exit | MakeArray(_) => target
  }
}

fn beta_contraction(c : Cps, optimized : Ref[Bool]) -> Cps {
  fn rec(c : Cps) {
    beta_contraction(c, optimized)
  }

  match c {
    Tuple(arr, bind, inner) => Tuple(arr, bind, rec(inner))
    KthTuple(idx, v, bind, inner) => KthTuple(idx, v, bind, rec(inner))
    Fix(f, args, body, rest) => {
      let body_info = beta_contract_info(body, f)
      let rest_info = beta_contract_info(rest, f)
      if body_info.num_called > 0 ||
        body_info.referred_as_closure ||
        rest_info.referred_as_closure ||
        rest_info.num_called > 1 {
        // NOTE: 
        // 1. self-recursive function won't beta extract
        // 2. any function referred as closure won't beta extract
        Fix(f, args, rec(body), rec(rest))
      } else {
        optimized.val = true
        let contracted = expand_function(f, args, body, rest)
        //println("=========================")
        //println("After contracting \{f}: ")
        //println(contracted)
        rec(contracted)
      }
    }
    If(cond, _then, _else) => If(cond, rec(_then), rec(_else))
    Prim(op, vs, bind, rest) => Prim(op, vs, bind, rec(rest))
    c => c
  }
}
