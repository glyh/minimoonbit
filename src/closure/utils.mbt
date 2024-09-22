fn forall[A](arr : Array[A], f : (A) -> Bool) -> Bool {
  arr.fold(init=true, fn(acc, ele) { acc && f(ele) })
}

fn zip2[A, B](arr1 : Array[A], arr2 : Array[B]) -> Array[(A, B)] {
  let out : Array[(A, B)] = []
  loop (arr1[:], arr2[:]) {
    ([], []) => break out
    ([a, .. as arr1], [b, .. as arr2]) => {
      out.push((a, b))
      continue (arr1, arr2)
    }
    _ => @util.die("zipping arrays of different size")
  }
}

enum Either[A, B] {
  Left(A)
  Right(B)
} derive(Show)
