typealias T = @types.Type

typealias Vid = @types.Vid

typealias TClass = @types.TClass

typealias TClasses = @types.TClasses

typealias TVar = @types.TVar

type TVarSet @immut/sorted_map.T[Vid, TClasses] derive(Show, Eq)

fn TVarSet::new() -> TVarSet {
  TVarSet(@immut/sorted_map.empty())
}

fn TVarSet::single(v : TVar) -> TVarSet {
  TVarSet(@immut/sorted_map.of([v]))
}

pub fn TVarSet::insert(
  self : TVarSet,
  vid : Vid,
  ~classes : TClasses = @immut/sorted_set.new()
) -> TVarSet {
  match self._.lookup(vid) {
    None => self._.insert(vid, classes)
    Some(classes_old) => self._.insert(vid, classes_old.union(classes))
  }
}

pub fn TVarSet::insert_var(self : TVarSet, v : TVar) -> TVarSet {
  self.insert(v.0, classes=v.1)
}

pub fn TVarSet::discard_var(self : TVarSet, v : TVar) -> TVarSet {
  self.discard(v.0)
}

pub fn TVarSet::discard(self : TVarSet, vid : Vid) -> TVarSet {
  self._.remove(vid)
}

pub fn TVarSet::contains(self : TVarSet, ele : Vid) -> Bool {
  self._.contains(ele)
}

pub fn TVarSet::op_add(self : TVarSet, rhs : TVarSet) -> TVarSet {
  rhs._.iter().fold(init=self, fn(acc, tup) { acc.insert_var(tup) })
}

pub fn TVarSet::op_sub(self : TVarSet, rhs : TVarSet) -> TVarSet {
  rhs._.iter().fold(init=self, fn(acc, tup) { acc.discard_var(tup) })
}

typealias Subst = @immut/sorted_map.T[Vid, T]

fn check_classes(t : T, clss : TClasses) -> Unit {
  fn helper(ty : T) -> (TClass) -> Unit {
    fn(cls) {
      match (ty, cls) {
        (Int | Double, Arith) => ()
        (Int | Double | Bool, Order) => ()
        (ty, cls) => @util.die("\{ty} doesn't implement typeclass \{cls}")
      }
    }
  }

  clss.each(helper(t))
}

typealias Constraint = (T, T)

typealias Symbol = String

typealias TyEnv = @immut/sorted_map.T[Symbol, T]
