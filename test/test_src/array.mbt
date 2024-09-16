fn array_map(len: Int, arr: Array[Int], f: (Int) -> Int) -> Array[Int] {
  fn loop1(i: Int) -> Unit {
    if 0 <= i {
      arr[i] = f(arr[i]); 
      loop1(i-1)
    }
  };
  loop1(len)
};

//fn array_fold(len: Int, arr: Array[Int], init: Int, f: (Int, Int) -> Int) -> Array[Int] {
//  fn loop()
//}
