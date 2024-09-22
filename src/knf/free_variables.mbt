type FVSet @hashset.T[Name]

fn FVSet::single(n : Name) -> FVSet {
  @hashset.of([n])
}

fn FVSet::empty() -> FVSet {
  @hashset.new()
}

fn FVSet::seq(ns : Iter[Name]) -> FVSet {
  let ret = @hashset.new()
  for n in ns {
    ret.insert(n)
  }
  ret
}

fn FVSet::op_add(lhs : FVSet, rhs : FVSet) -> FVSet {
  lhs._.union(rhs._)
}

fn FVSet::op_sub(lhs : FVSet, rhs : FVSet) -> FVSet {
  lhs._.difference(rhs._)
}

fn FVSet::discard(self : FVSet, target : Name) -> FVSet {
  self._.remove(target)
  self
}

fn FVSet::discard_many(self : FVSet, targets : Iter[Name]) -> FVSet {
  self._.difference(@hashset.from_iter(targets))
}

pub fn FVSet::is_empty(self : FVSet) -> Bool {
  self._.is_empty()
}

pub fn FVSet::iter(self : FVSet) -> Iter[Name] {
  self._.iter()
}

pub fn FVSet::contains(self : FVSet, v : Name) -> Bool {
  self._.contains(v)
}

pub fn FuncDef::free_variables(self : FuncDef) -> FVSet {
  self.body.free_variables() -
  FVSet::single(self.name) -
  FVSet::seq(self.args.iter().map(fn { (id, _) => id }))
}

pub fn Knf::free_variables(self : Knf) -> FVSet {
  match self {
    Unit | Int(_) | Double(_) => FVSet::empty()
    Var(name) | Neg(name) | FNeg(name) | ExternalArray(name) =>
      FVSet::single(name)
    Tuple(tup) => FVSet::seq(tup.iter())
    Apply(f, args) => FVSet::single(f) + FVSet::seq(args.iter())
    Get(lhs, rhs)
    | Add(lhs, rhs)
    | Sub(lhs, rhs)
    | Mul(lhs, rhs)
    | Div(lhs, rhs)
    | FAdd(lhs, rhs) | FSub(lhs, rhs) | FMul(lhs, rhs) | FDiv(lhs, rhs) =>
      FVSet::seq([lhs, rhs].iter())
    IfEq(lhs, rhs, _then, _else) | IfLe(lhs, rhs, _then, _else) =>
      FVSet::seq([lhs, rhs].iter()) +
      _then.free_variables() +
      _else.free_variables()
    Let((name, _), rhs, rest) =>
      rest.free_variables().discard(name) + rhs.free_variables()
    LetRec(f, rest) =>
      rest.free_variables().discard(f.name) +
      f.body
      .free_variables()
      .discard(f.name)
      .discard_many(f.args.iter().map(fn { (arg, _) => arg }))
    LetTuple(tup, rhs, rest) =>
      rest.free_variables() -
      FVSet::seq(tup.iter().map(fn { (name, _) => name })) +
      FVSet::single(rhs)
    Put(arr, idx, rhs) => FVSet::seq([arr, idx, rhs].iter())
    ExternalFunctionApplication(_, args) => FVSet::seq(args.iter())
  }
}
