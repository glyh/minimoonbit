let x: Int = 9 + 1;

fn f(y: Int) -> Int {
  if y == 0 {
    0
  } else {
    x + f(y - 1)
  }
};

fn main {
  print_int(f(12))
};
