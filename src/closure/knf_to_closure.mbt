pub fn knf_program_to_closure(
  k : K,
  external_env : Map[String, Type]
) -> Program {
  let env = CloEnv::new(external_env, k)
  let body = env.collect_closure(k)
  let fundefs = env.fundefs.iter().map(fn { (_, v) => v }).collect()
  { fundefs, body }
}
