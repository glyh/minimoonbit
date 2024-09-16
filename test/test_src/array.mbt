fn array_map(len: Int, arr: Array[Int], f: (Int) -> Int) -> Array[Int] {
  fn loop1(i: Int) -> Unit {
    if 0 <= i {
      arr[i] = f(arr[i]); 
      loop1(i-1)
    }
  };
  let _ = loop1(len);
  arr
};

fn array_foldr(len: Int, arr: Array[Int], init: Int, f: (Int, Int) -> Int) -> Array[Int] {
  let out = Array::make(1, init);
  fn loop1(i: Int) -> Unit {
    if 0 <= i {
      out[i] = f(out[i], arr[i]);
      loop1(i-1)
    }
  };
  let _ = loop1(len);
  arr[0]
};

fn array_filt_in_place(len: Int, arr: Array[Int], init: Int, f: (Int) -> Bool) -> Int {
  let ptrs = Array::make(2, 0);
  fn loop1(i: Int) -> Unit {
    let cond = Array::make(1, false);
    if ptrs[1] <= len - 1 {
      let _ = if f(arr[ptrs[1]]) {
        arr[ptrs[0]] = arr[ptrs[1]];
        ptrs[0] = ptrs[0] + 1;
        ptrs[1] = ptrs[1] + 1;
        ()
      } else {
        ptrs[1] = ptrs[1] + 1;
        ()
      };
      loop1(i - 1)
    }
  };
  let _ = loop1(len);
  ptrs[0]
};

fn array_zip(len: Int, arr1: Array[Int], arr2: Array[Int]) -> Array[(Int, Int)] {
  let result = Array::make(len, (0, 0));
  fn loop1(i: Int) -> Unit {
    if 0 <= i {
      result[i] = (arr1[i], arr2[i]);
      loop1(i - 1)
    }
  };
  let _ = loop1(len);
  result
};

fn currify(f: (Int, Int, Int, Int) -> Int) -> (Int) -> (Int) -> (Int) -> (Int) -> Int {
  let result = {fn _1(i) {
    fn _2(j) {
      fn _3(k) {
        fn _4(l) {
          f(i, j, k, l)
        }; _4
      }; _3
    }; _2
  }; _1};
  result
};

fn ping_pong() -> (Int) -> Unit {
  fn get_ping(pong){
    fn ping(i) {
      pong(i-1)
    }; ping
  };
  fn pong(i) {
    get_ping(pong)(i-1)
  }; pong
};
