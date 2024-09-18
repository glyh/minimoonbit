pub typealias Vid = Int

pub enum TClass {
  Arith
  Order
} derive(Show, Eq, Compare)

pub typealias TClasses = @immut/sorted_set.T[TClass]

pub typealias TVar = (Vid, TClasses)

let counter : Ref[Vid] = { val: 0 }

fn tvar_helper(classes : TClasses) -> Type {
  counter.val = counter.val + 1
  let var_ref : Ref[Result[Type, TVar]] = { val: Err((counter.val, classes)) }
  Var(var_ref)
}

pub fn dummy_tvar() -> Type {
  tvar_helper(@immut/sorted_set.new())
}

pub fn classed_tvar(classes : FixedArray[TClass]) -> Type {
  tvar_helper(@immut/sorted_set.of(classes))
}
