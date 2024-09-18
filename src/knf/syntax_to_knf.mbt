// REF: https://esumii.github.io/min-caml/tutorial-mincaml-9.eng.htm
pub typealias LocalEnv = @immut/hashmap.T[String, Type]

struct KnfEnv {
  mut counter : Int
  externals : @immut/hashmap.T[String, Type]
}

pub fn KnfEnv::new(externals : @immut/hashmap.T[String, Type]) -> KnfEnv {
  { counter: 1, externals }
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

fn KnfEnv::new_named_temp(self : KnfEnv, name : Name) -> Name {
  let counter = self.counter
  self.counter += 1
  { ..name, slot: counter }
}

//fn KnfEnv::new_named(self : KnfEnv, name : String) -> Name {
//  let counter = self.counter
//  self.counter += 1
//  { name: Some(name), slot: counter }
//}
//

pub fn KnfEnv::to_knf(self : KnfEnv, s : S) -> Knf {
  let rec = fn { s => self.to_knf(s) }
  match s {
    Unit => Unit
    Bool(true) => Int(1)
    Bool(false) => Int(0)
    Int(i) => Int(i)
    Double(f) => Double(f)
    //Var(name) => Var(self.new_named(name))
    //Tuple(tup) => tup.fold
  }
}
