pub struct Var {
  name : String?
  id : Int
} derive(Eq, Hash, Compare)

impl Show for Var with output(self, logger) {
  logger.write_string(self.to_string())
}

pub fn Var::to_string(self : Var) -> String {
  match self.name {
    None => "?\{self.id}"
    Some(n) => if self.id < 0 { n } else { "\{n}.\{self.id}" }
  }
}

pub fn Var::var_of_external(ext_name : String) -> Var {
  { name: Some(ext_name), id: -1 }
}
