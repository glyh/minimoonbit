pub let externals_list : Array[(String, Type)] = [
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
  ("create_array", Fun([Int], Array(Int))),
  ("create_float_array", Fun([Int], Array(Double))),
  ("create_ptr_array", Fun([Int], Array(Ptr))),
]

pub let externals : @immut/hashmap.T[String, Type] = externals_list.iter()
  |> @immut/hashmap.from_iter()

pub let externals_idx : @hashmap.T[String, Int] = externals_list
  .mapi(
    fn(idx, item) {
      let (name, _) = item
      (name, -idx - 1)
    },
  )
  .iter()
  |> @hashmap.from_iter()
