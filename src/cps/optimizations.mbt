pub fn optimize_cps(c : Cps) -> Cps {
  let mut c = c
  let optimized = { val: true }
  while optimized.val {
    optimized.val = false
    c = beta_contraction(c, optimized)
    //if optimized.val {
    //  println("beta_contract")
    //}
  }
  optimized.val = true
  while optimized.val {
    optimized.val = false
    c = alias_analysis(c, optimized)
    //if optimized.val {
    //  println("alias_analysis")
    //}
  }
  c
}
