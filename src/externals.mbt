pub let externals : @immut/hashmap.T[String, Type] = @immut/hashmap.of(
  [
    ("read_int", Fun([], Int)),
    ("print_int", Fun([Int], Unit)),
    ("read_char", Fun([], Int)),
    ("print_char", Fun([Int], Unit)),
    ("print_endline", Fun([], Unit)),
    ("int_of_float", Fun([Double], Int)),
    ("float_of_int", Fun([Int], Double)),
    ("truncate", Fun([Double], Int)),
    ("floor", Fun([Double], Double)),
    ("abs_float", Fun([Double], Double)),
    ("abs_float", Fun([Double], Double)),
    ("sqrt", Fun([Double], Double)),
    ("sin", Fun([Double], Double)),
    ("cos", Fun([Double], Double)),
    ("atan", Fun([Double], Double)),
  ],
)
