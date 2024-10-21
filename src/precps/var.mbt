pub struct Var {
  name : String?
  id : Int
} derive(Eq, Hash, Compare)

impl Show for Var with output(self, logger) {
  logger.write_string(self.to_string())
}

pub fn Var::to_string(self : Var) -> String {
  match self.name {
    None => "_minimbt_\{self.id}"
    Some(n) => if self.id < 0 { n } else { "_minimbt_\{n}_\{self.id}" }
  }
}

pub fn Var::var_of_external(ext_name : String) -> Var {
  { name: Some(ext_name), id: @types.externals_idx[ext_name].unwrap() }
}
