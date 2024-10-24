pub fn optimize_cps(c : Cps) -> Cps {
  c |> beta_contraction()
}
