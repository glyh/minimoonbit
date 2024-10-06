pub struct CpsEnv {
  counter : Ref[Int]
}

pub fn CpsEnv::new(counter : Int) -> CpsEnv {
  { counter: { val: counter } }
}

fn CpsEnv::new_tmp(self : CpsEnv, t : T) -> Var {
  self.counter.val = self.counter.val + 1
  { name: { val: None }, id: self.counter.val, ty: t }
}

fn CpsEnv::new_named(self : CpsEnv, s : String, t : T) -> Var {
  self.counter.val = self.counter.val + 1
  { name: { val: Some(s) }, id: self.counter.val, ty: t }
}
