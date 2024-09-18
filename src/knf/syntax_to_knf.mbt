// REF: 
// 1. https://esumii.github.io/min-caml/tutorial-mincaml-9.eng.htm
// 2. https://www.cs.tufts.edu/comp/150VM/modules/09Aknormalize.html
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

fn KnfEnv::new_var(self : KnfEnv, name : String) -> Name {
  let counter = self.counter
  self.counter += 1
  { name: Some(name), slot: counter }
}

enum DefoldKnf {
  Let(Name, Type, Knf)
  LetRec(FuncDef)
  LetTuple(Array[(Name, Type)], Name)
}

fn refold_knf(init : Knf, defolds : Array[DefoldKnf]) -> Knf {
  fn fold1(acc : Knf, ele : DefoldKnf) -> Knf {
    match ele {
      Let(name, ty, rhs) => Let((name, ty), rhs, acc)
      LetRec(f) => LetRec(f, acc)
      LetTuple(tup, rhs) => LetTuple(tup, rhs, acc)
    }
  }

  defolds.rev_fold(~init, fold1)
}

typealias TyEnv = @immut/hashmap.T[String, T]

pub fn to_knf_with_tenv(kenv : KnfEnv, tyenv : TyEnv, s : S) -> (Knf, T) {
  match s {
    Unit => (Unit, Unit)
    Bool(true) => (Int(1), Bool)
    Bool(false) => (Int(0), Bool)
    Int(i) => (Int(i), Int)
    Double(f) => (Double(f), Double)
    Var(name) =>
      match tyenv.find(name) {
        None => @util.die("Can't find type for \{name} during KNF conversion")
        Some(ty) => (Var(kenv.new_var(name)), ty)
      }
    Tuple(tup) => {
      let names : Array[N] = []
      let heads : Array[DefoldKnf] = []
      let tys : Array[T] = []
      for ti in tup.iter() {
        let ki = to_knf_with_tenv(kenv, tyenv, ti)
        let name = kenv.new_temp()
        names.push(name)
        heads.push(Let(name, ki.1, ki.0))
        tys.push(ki.1)
      }
      (refold_knf(Tuple(names), heads), Tuple(tys))
    }
    //Not(b) => ()
  }
}

pub fn KnfEnv::to_knf(self : KnfEnv, s : S) -> Knf {
  to_knf_with_tenv(self, @immut/hashmap.new(), s).0
}
