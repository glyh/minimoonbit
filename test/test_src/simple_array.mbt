fn main {
  let a = Array::make(3, 0);
  a[1] = 3;
  a[2] = 2;
  a[0] = a[1] + a[2];
  let _ = print_int(a[0]);
  print_endline()
};
