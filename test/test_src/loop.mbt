fn loop(x: Int) -> Int {
  if x <= 1 {
    x
  } else {
    loop(x-1) + loop(x-3)
  }
};

fn main {
  print_int(loop(10))
};
