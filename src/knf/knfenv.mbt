struct KnfEnv {
  mut counter : Int
  init_externals_name : @hashset.T[String]
  externals : @immut/hashmap.T[String, T]
  nonexternals : @immut/hashmap.T[Name, T]
}

pub fn KnfEnv::new(externals : @immut/hashmap.T[String, T]) -> KnfEnv {
  let externals = externals.iter().map(fn { (k, v) => ("minimbt_" + k, v) })
    |> @immut/hashmap.from_iter()
  {
    counter: 1,
    externals,
    nonexternals: @immut/hashmap.new(),
    init_externals_name: externals.iter().map(fn { (k, _) => k })
    |> @hashset.from_iter(),
  }
}

pub fn KnfEnv::add(self : KnfEnv, name : Name, ty : T) -> KnfEnv {
  match name {
    { slot: 0, name: Some(name) } =>
      { ..self, externals: self.externals.add(name, ty) }
    name => { ..self, nonexternals: self.nonexternals.add(name, ty) }
  }
}

pub fn KnfEnv::has_external(self : KnfEnv, name : String) -> Bool {
  match self.externals.find(name) {
    Some(_) => true
    _ => false
  }
}

pub fn KnfEnv::find(self : KnfEnv, name : Name) -> T? {
  match name {
    { slot: 0, name: Some(name) } => self.externals.find(name)
    name => self.nonexternals.find(name)
  }
}

pub fn KnfEnv::init_counter_from_existing(
  self : KnfEnv,
  existing : Knf
) -> Unit {
  self.counter = knf_max_counter(existing) + 1
}

fn KnfEnv::new_temp(self : KnfEnv) -> Name {
  let temp = Name::slot_only(self.counter)
  self.counter += 1
  temp
}

fn KnfEnv::new_name(self : KnfEnv, name : String) -> Name {
  if self.init_externals_name.contains("minimbt_" + name) {
    // if this name is potentially external, we add a prefix
    { name: Some("minimbt_" + name), slot: 0 }
  } else {
    { name: Some(name), slot: 0 }
  }
}
