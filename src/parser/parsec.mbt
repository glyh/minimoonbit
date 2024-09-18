// REF: https://moonbitlang.github.io/moonbit-textbook/parser/#syntax-parsing

typealias Lexeme = (@lex.Token, @lex.Loc, @lex.Loc)

type Parser[V] (ArrayView[Lexeme]) -> (V?, ArrayView[Lexeme])

fn parse[T](
  self : Parser[T],
  input : ArrayView[Lexeme]
) -> (T?, ArrayView[Lexeme]) {
  (self._)(input)
}

fn empty[T]() -> Parser[T] {
  fn { view => (None, view) }
}

fn epsilon() -> Parser[Unit] {
  fn { view => (Some(()), view) }
}

fn ref[T](referred : Ref[Parser[T]]) -> Parser[T] {
  fn { input => referred.val.parse(input) }
}

let double : Parser[Double] = fn {
  [(@lex.F64(f), _, _), .. as rest] => (Some(f), rest)
  view => (None, view)
}

let int : Parser[Int] = fn {
  [(@lex.I32(i), _, _), .. as rest] => (Some(i), rest)
  view => (None, view)
}

let bool : Parser[Bool] = fn {
  [(@lex.BOOL(b), _, _), .. as rest] => (Some(b), rest)
  view => (None, view)
}

let id : Parser[String] = fn {
  [(@lex.ID(i), _, _), .. as rest] => (Some(i), rest)
  view => (None, view)
}

fn id_p(p : (String) -> Bool) -> Parser[String] {
  fn {
    [(@lex.ID(id), _, _), .. as rest] as view =>
      if p(id) {
        (Some(id), rest)
      } else {
        (None, view)
      }
    view => (None, view)
  }
}

let line_start : Parser[Unit] = fn {
  [(_, start_loc, _), ..] as view =>
    if start_loc.col == 1 {
      (Some(()), view)
    } else {
      (None, view)
    }
  view => (None, view)
}

fn op_add[T](self : Parser[T], rhs : Parser[T]) -> Parser[T] {
  self.or(rhs)
}

fn or[T](self : Parser[T], rhs : Parser[T]) -> Parser[T] {
  Parser(
    fn(input) {
      match self.parse(input) {
        (None, _) => rhs.parse(input)
        result => result
      }
    },
  )
}

fn lift2[U, V, W](f : (U, V) -> W, p1 : Parser[U], p2 : Parser[V]) -> Parser[W] {
  Parser(
    fn(input) {
      match p1.parse(input) {
        (None, _) => (None, input)
        (Some(u), rest) =>
          match p2.parse(rest) {
            (None, _) => (None, input)
            (Some(v), rest2) => (Some(f(u, v)), rest2)
          }
      }
    },
  )
}

fn lift2_maybe[U, V, W](
  f : (U, V) -> W?,
  p1 : Parser[U],
  p2 : Parser[V]
) -> Parser[W] {
  Parser(
    fn(input) {
      match p1.parse(input) {
        (None, _) => (None, input)
        (Some(u), rest) =>
          match p2.parse(rest) {
            (None, _) => (None, input)
            (Some(v), rest2) => (f(u, v), rest2)
          }
      }
    },
  )
}

fn lift3[U, V, W, X](
  f : (U, V, W) -> X,
  p1 : Parser[U],
  p2 : Parser[V],
  p3 : Parser[W]
) -> Parser[X] {
  Parser(
    fn(input) {
      match p1.parse(input) {
        (None, _) => (None, input)
        (Some(u), rest) =>
          match p2.parse(rest) {
            (None, _) => (None, input)
            (Some(v), rest2) =>
              match p3.parse(rest2) {
                (None, _) => (None, input)
                (Some(w), rest3) => (Some(f(u, v, w)), rest3)
              }
          }
      }
    },
  )
}

fn lift3_maybe[U, V, W, X](
  f : (U, V, W) -> X?,
  p1 : Parser[U],
  p2 : Parser[V],
  p3 : Parser[W]
) -> Parser[X] {
  Parser(
    fn(input) {
      match p1.parse(input) {
        (None, _) => (None, input)
        (Some(u), rest) =>
          match p2.parse(rest) {
            (None, _) => (None, input)
            (Some(v), rest2) =>
              match p3.parse(rest2) {
                (None, _) => (None, input)
                (Some(w), rest3) => (f(u, v, w), rest3)
              }
          }
      }
    },
  )
}

fn lift4[U, V, W, X, Y](
  f : (U, V, W, X) -> Y,
  p1 : Parser[U],
  p2 : Parser[V],
  p3 : Parser[W],
  p4 : Parser[X]
) -> Parser[Y] {
  Parser(
    fn(input) {
      match p1.parse(input) {
        (None, _) => (None, input)
        (Some(u), rest) =>
          match p2.parse(rest) {
            (None, _) => (None, input)
            (Some(v), rest2) =>
              match p3.parse(rest2) {
                (None, _) => (None, input)
                (Some(w), rest3) =>
                  match p4.parse(rest3) {
                    (None, _) => (None, input)
                    (Some(x), rest4) => (Some(f(u, v, w, x)), rest4)
                  }
              }
          }
      }
    },
  )
}

fn map[U, V](self : Parser[U], f : (U) -> V) -> Parser[V] {
  Parser(
    fn(input) {
      match self.parse(input) {
        (None, rest) => (None, rest)
        (Some(u), rest) => (Some(f(u)), rest)
      }
    },
  )
}

fn op_shr[U, V](self : Parser[U], rhs : Parser[V]) -> Parser[V] {
  Parser(
    fn(input) {
      match self.parse(input) {
        (None, _) => (None, input)
        (Some(_), rest) => rhs.parse(rest)
      }
    },
  )
}

fn op_shl[U, V](self : Parser[U], rhs : Parser[V]) -> Parser[U] {
  Parser(
    fn(input) {
      match self.parse(input) {
        (None, rest) => (None, rest)
        (Some(u), rest) =>
          match rhs.parse(rest) {
            (None, _) => (None, input)
            (Some(_), rest2) => (Some(u), rest2)
          }
      }
    },
  )
}

fn tok(tok : @lex.Token) -> Parser[@lex.Token] {
  Parser(
    fn(input) {
      match input {
        [(tok2, _, _), .. as rest] =>
          if tok == tok2 {
            (Some(tok), rest)
          } else {
            (None, input)
          }
        _ => (None, input)
      }
    },
  )
}

fn maybe[U](p : Parser[U]) -> Parser[U?] {
  Parser(
    fn(input) {
      match p.parse(input) {
        (Some(u), rest) => (Some(Some(u)), rest)
        (None, rest) => (Some(None), rest)
      }
    },
  )
}

fn many[U](inner : Parser[U]) -> Parser[Array[U]] {
  Parser(
    fn(input) {
      let acc : Array[U] = []
      let mut input = input
      while true {
        match inner.parse(input) {
          (Some(v), rest) => {
            acc.push(v)
            input = rest
          }
          (None, _) => break
        }
      }
      (Some(acc), input)
    },
  )
}

fn sep_list1[U, V](sep : Parser[U], inner : Parser[V]) -> Parser[Array[V]] {
  Parser(
    fn(input) {
      match inner.parse(input) {
        (Some(i), rest) => {
          let acc : Array[V] = [i]
          let mut rest = rest
          let sep_in = sep >> inner
          while true {
            match sep_in.parse(rest) {
              (None, _) => break
              (Some(i), rest2) => {
                rest = rest2
                acc.push(i)
              }
            }
          }
          (Some(acc), rest)
        }
        (None, _) => (None, input)
      }
    },
  )
}

fn sep_list[U, V](sep : Parser[U], inner : Parser[V]) -> Parser[Array[V]] {
  Parser(
    fn(input) {
      match inner.parse(input) {
        (Some(i), rest) => {
          let acc : Array[V] = [i]
          let mut rest = rest
          let sep_in = sep >> inner
          while true {
            match sep_in.parse(rest) {
              (None, _) => break
              (Some(i), rest2) => {
                rest = rest2
                acc.push(i)
              }
            }
          }
          (Some(acc), rest)
        }
        (None, _) => (Some([]), input)
      }
    },
  )
}

fn fold_right[U, V](
  init : Parser[V],
  ele : Parser[U],
  op : Parser[(U, V) -> V]
) -> Parser[V] {
  let prefix = lift2(fn(ele, op) { (ele, op) }, ele, op)
  lift2(
    fn(rest_r : Array[(U, (U, V) -> V)], last : V) {
      rest_r.rev_fold(
        init=last,
        fn(acc, prefix) {
          let (lhs, op) = prefix
          op(lhs, acc)
        },
      )
    },
    many(prefix),
    init,
  )
}

fn replace[U, V](self : Parser[U], v : V) -> Parser[V] {
  Parser(
    fn(input) {
      match self.parse(input) {
        (None, _) => (None, input)
        (Some(_), rest) => (Some(v), rest)
      }
    },
  )
}

fn fold_left[U, V](
  init : Parser[U],
  ele : Parser[V],
  op : Parser[(U, V) -> U]
) -> Parser[U] {
  let postfix = lift2(fn(op, ele) { (op, ele) }, op, ele)
  lift2(
    fn(first : U, rest : Array[((U, V) -> U, V)]) {
      rest.fold(
        init=first,
        fn(acc, postfix) {
          let (op, rhs) = postfix
          op(acc, rhs)
        },
      )
    },
    init,
    many(postfix),
  )
}
