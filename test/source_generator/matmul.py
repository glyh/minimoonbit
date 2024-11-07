#!/usr/bin/env python
# Multiply a n by m matrix and a m by k matrix 
import random
from typing import List

[l, n, m] = [10, 10, 10]

def generate_mat(m: int, n: int, min_value: float = -10.0, max_value: float = 10.0) -> List[List[float]]:
    """
    Generates an m x n matrix filled with random floating-point numbers.
    
    :param m: Number of rows
    :param n: Number of columns
    :param min_value: Minimum value for random floats (inclusive)
    :param max_value: Maximum value for random floats (inclusive)
    :return: m x n matrix with random floating-point numbers
    """
    return [[round(random.uniform(min_value, max_value), 2) for _ in range(n)] for _ in range(m)]


indent = 2
indent_str = " " * indent

def format_mat(symbol: str, m: int, n: int, mat: List[List[float]]) -> str:
    ret = ""
    for row in range(m):
        for col in range(n):
            ret += indent_str if col == 0 else "; "
            ret += f"{symbol}[{row}][{col}] = {mat[row][col]}"
        ret += ";\n"

    return ret


operate_mat = \
f"""
  let a = gen_arr({l}, {n});
  let b = gen_arr({n}, {m});
  let c = gen_arr({l}, {m});
"""

mat_a = generate_mat(l, n)
operate_mat += "\n" + format_mat("a", l, n, mat_a)

mat_b = generate_mat(n, m)
operate_mat += "\n" + format_mat("b", n, m, mat_b)

operate_mat += "\n" + indent_str + f"let _ = matmul({l},{n},{m},a,b,c);"


operate_mat += "\n" + indent_str + f"matshow({l}, {m}, c)"

operate_mat += "\n"

generated = """
fn matshow(m: Int, n: Int, mat: Array[Array[Double]]) -> Unit {
  fn loop1(i: Int) -> Unit {
    if i <= m - 1 {
      fn loop2(j: Int) -> Unit {
        if j <= n - 1 {
          let _ = print_int(truncate(mat[i][j]));
          let _ = print_char(32);
          loop2(j+1)
        } else {
          print_endline()
        }
      }; 
      let _ = loop2(0);
      loop1(i+1)
    } else {
      ()
    }
  }; 
  loop1(0)
};

fn matmul(l: Int, m: Int, n: Int, a: Array[Array[Double]], b: Array[Array[Double]], c: Array[Array[Double]]) -> Unit {
  fn loop1(i: Int) -> Unit {
    if 0 <= i {
      fn loop2(j: Int) -> Unit {
        if 0 <= j {
          fn loop3(k: Int) -> Unit {
            if 0 <= k {
              c[i][j] = c[i][j] + a[i][k] * b[k][j];
              loop3(k - 1)
            } else {
              ()
            }
          };
          let _ = loop3(m - 1);
          loop2(j - 1)
        } else {
          ()
        }
      };
      let _ = loop2(n - 1);
      loop1(i - 1)
    } else {
      ()
    }
  };
  loop1(l - 1)
};

fn main {
  let dummy = Array::make(0, 0.0);
  fn gen_arr(m: Int, n: Int) -> Array[Array[Double]] {
    let mat = Array::make(m, dummy);
    fn init_arr(i: Int) -> Unit {
      if 0 <= i {
        mat[i] = Array::make(n, 0.0);
        init_arr(i - 1)
      } else {
        ()
      }
    };
    let _ = init_arr(m - 1);
    mat
  };
""" \
 + operate_mat + \
"""
};"""

source_path = "../test_src/matmul_gen.mbt"
ans_path = "../test_src/matmul_gen.ans"

with open(source_path, 'w') as source_file:
    source_file.write(generated)

def matrix_multiply(matrix_a, matrix_b):
    # Get the dimensions of the matrices
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    # Check if multiplication is possible
    if cols_a != rows_b:
        raise ValueError("Cannot multiply: The number of columns in matrix A must equal the number of rows in matrix B.")

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    # Perform multiplication
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]

    return result

def float_to_ieee754(float_value):
    import struct
    # Pack the float as a binary string (double precision)
    packed = struct.pack('>d', float_value)  # '>d' means big-endian double
    # Convert the packed binary string to an integer representation
    integer_representation = int.from_bytes(packed, byteorder='big')
    return integer_representation

mat_c = matrix_multiply(mat_a, mat_b)

with open(ans_path, 'w') as ans_file:
    for row in mat_c:
        for ele in row:
            ans_file.write(f'{int(ele)} ')
        ans_file.write('\n')


