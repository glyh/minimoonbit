fn loop(x: Int) -> Int {
  if x <= 1 {
    x
  } else {
    loop(x-1)
  }
};

fn main {
  print_int(loop(10))
};
