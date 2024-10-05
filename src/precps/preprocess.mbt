pub fn inline_entry(
  s : S,
  ~has_main : Bool = false,
  ~has_init : Bool = false
) -> S {
  match s {
    Unit =>
      match (has_main, has_init) {
        (false, false) => @util.die("no entrance found")
        (true, false) => App(Var("main"), [])
        (false, true) => App(Var("init"), [])
        (true, true) =>
          Let(("_", Unit), App(Var("init"), []), App(Var("main"), []))
      }
    LetRec(f, rest) => {
      let has_main = has_main || f.name.0 == "main"
      let has_init = has_init || f.name.0 == "init"
      LetRec(f, inline_entry(rest, ~has_main, ~has_init))
    }
    s => @util.die("unexpected toplevel \{s}")
  }
}
