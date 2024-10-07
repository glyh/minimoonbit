fn zip2[A : Show, B : Show](arr1 : Array[A], arr2 : Array[B]) -> Array[(A, B)] {
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
