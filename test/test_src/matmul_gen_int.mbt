
fn matshow(m: Int, n: Int, mat: Array[Array[Int]]) -> Unit {
  fn loop1(i: Int) -> Unit {
    if i <= m - 1 {
      fn loop2(j: Int) -> Unit {
        if j <= n - 1 {
          let _ = print_int(mat[i][j]);
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

fn matmul(l: Int, m: Int, n: Int, a: Array[Array[Int]], b: Array[Array[Int]], c: Array[Array[Int]]) -> Unit {
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
  let dummy = Array::make(0, 0);
  fn gen_arr(m: Int, n: Int) -> Array[Array[Int]] {
    let mat = Array::make(m, dummy);
    fn init_arr(i: Int) -> Unit {
      if 0 <= i {
        mat[i] = Array::make(n, 0);
        init_arr(i - 1)
      } else {
        ()
      }
    };
    let _ = init_arr(m - 1);
    mat
  };

  let a = gen_arr(23, 22);
  let b = gen_arr(22, 21);
  let c = gen_arr(23, 21);

  a[0][0] = -8; a[0][1] = -5; a[0][2] = 4; a[0][3] = -8; a[0][4] = 4; a[0][5] = -3; a[0][6] = -2; a[0][7] = 6; a[0][8] = 0; a[0][9] = -3; a[0][10] = -5; a[0][11] = 4; a[0][12] = 8; a[0][13] = 4; a[0][14] = 1; a[0][15] = 5; a[0][16] = -2; a[0][17] = -1; a[0][18] = 2; a[0][19] = 5; a[0][20] = -7; a[0][21] = 1;
  a[1][0] = -4; a[1][1] = -4; a[1][2] = -4; a[1][3] = 1; a[1][4] = 7; a[1][5] = 1; a[1][6] = -6; a[1][7] = -8; a[1][8] = -8; a[1][9] = -6; a[1][10] = 1; a[1][11] = 6; a[1][12] = 7; a[1][13] = -5; a[1][14] = 0; a[1][15] = 1; a[1][16] = 0; a[1][17] = -1; a[1][18] = 4; a[1][19] = 8; a[1][20] = -6; a[1][21] = -4;
  a[2][0] = -1; a[2][1] = 6; a[2][2] = 8; a[2][3] = 1; a[2][4] = -1; a[2][5] = 5; a[2][6] = -5; a[2][7] = -4; a[2][8] = -7; a[2][9] = 2; a[2][10] = -6; a[2][11] = 8; a[2][12] = -1; a[2][13] = -5; a[2][14] = -8; a[2][15] = -7; a[2][16] = 0; a[2][17] = -4; a[2][18] = -4; a[2][19] = -6; a[2][20] = 2; a[2][21] = -5;
  a[3][0] = -7; a[3][1] = -8; a[3][2] = 6; a[3][3] = -8; a[3][4] = -5; a[3][5] = 0; a[3][6] = -7; a[3][7] = 3; a[3][8] = 0; a[3][9] = -6; a[3][10] = -3; a[3][11] = -8; a[3][12] = 1; a[3][13] = 1; a[3][14] = 1; a[3][15] = 3; a[3][16] = 8; a[3][17] = -3; a[3][18] = 0; a[3][19] = -4; a[3][20] = 2; a[3][21] = 2;
  a[4][0] = -5; a[4][1] = 8; a[4][2] = 1; a[4][3] = -5; a[4][4] = -3; a[4][5] = 0; a[4][6] = 5; a[4][7] = 2; a[4][8] = -7; a[4][9] = -1; a[4][10] = -4; a[4][11] = -4; a[4][12] = -5; a[4][13] = -3; a[4][14] = -2; a[4][15] = 2; a[4][16] = -7; a[4][17] = 6; a[4][18] = 7; a[4][19] = 5; a[4][20] = 7; a[4][21] = 7;
  a[5][0] = 5; a[5][1] = -6; a[5][2] = 4; a[5][3] = 6; a[5][4] = 8; a[5][5] = -3; a[5][6] = -3; a[5][7] = -6; a[5][8] = -2; a[5][9] = -2; a[5][10] = 0; a[5][11] = -6; a[5][12] = 2; a[5][13] = 0; a[5][14] = -7; a[5][15] = 8; a[5][16] = -5; a[5][17] = -3; a[5][18] = -8; a[5][19] = -2; a[5][20] = 6; a[5][21] = 3;
  a[6][0] = -4; a[6][1] = 5; a[6][2] = -1; a[6][3] = 1; a[6][4] = 3; a[6][5] = 4; a[6][6] = -3; a[6][7] = -3; a[6][8] = 7; a[6][9] = -8; a[6][10] = -4; a[6][11] = -4; a[6][12] = 4; a[6][13] = 8; a[6][14] = 4; a[6][15] = 7; a[6][16] = 4; a[6][17] = 4; a[6][18] = 2; a[6][19] = 0; a[6][20] = 1; a[6][21] = 2;
  a[7][0] = -4; a[7][1] = 3; a[7][2] = 6; a[7][3] = 6; a[7][4] = -1; a[7][5] = -6; a[7][6] = -3; a[7][7] = 2; a[7][8] = 0; a[7][9] = -1; a[7][10] = 1; a[7][11] = 4; a[7][12] = 0; a[7][13] = 8; a[7][14] = 4; a[7][15] = -6; a[7][16] = 5; a[7][17] = 0; a[7][18] = -2; a[7][19] = 1; a[7][20] = -8; a[7][21] = -8;
  a[8][0] = -1; a[8][1] = 8; a[8][2] = -7; a[8][3] = 0; a[8][4] = 1; a[8][5] = 1; a[8][6] = -5; a[8][7] = 5; a[8][8] = 5; a[8][9] = -8; a[8][10] = -4; a[8][11] = -7; a[8][12] = -7; a[8][13] = 8; a[8][14] = 2; a[8][15] = 7; a[8][16] = 8; a[8][17] = 4; a[8][18] = 2; a[8][19] = 3; a[8][20] = 8; a[8][21] = 1;
  a[9][0] = 5; a[9][1] = -8; a[9][2] = 6; a[9][3] = -2; a[9][4] = 3; a[9][5] = -8; a[9][6] = -1; a[9][7] = -4; a[9][8] = 6; a[9][9] = -6; a[9][10] = -6; a[9][11] = -4; a[9][12] = -6; a[9][13] = -3; a[9][14] = -6; a[9][15] = -3; a[9][16] = 5; a[9][17] = 0; a[9][18] = -7; a[9][19] = 2; a[9][20] = 4; a[9][21] = -6;
  a[10][0] = 2; a[10][1] = -3; a[10][2] = 5; a[10][3] = -6; a[10][4] = -8; a[10][5] = -6; a[10][6] = -1; a[10][7] = -8; a[10][8] = 3; a[10][9] = 1; a[10][10] = 3; a[10][11] = 6; a[10][12] = -4; a[10][13] = -4; a[10][14] = 1; a[10][15] = 8; a[10][16] = 7; a[10][17] = 8; a[10][18] = 2; a[10][19] = -5; a[10][20] = 4; a[10][21] = -7;
  a[11][0] = -1; a[11][1] = -7; a[11][2] = -3; a[11][3] = 4; a[11][4] = -8; a[11][5] = -8; a[11][6] = 8; a[11][7] = -2; a[11][8] = 6; a[11][9] = -3; a[11][10] = -8; a[11][11] = -5; a[11][12] = 6; a[11][13] = 8; a[11][14] = -3; a[11][15] = -8; a[11][16] = 8; a[11][17] = -3; a[11][18] = -3; a[11][19] = -3; a[11][20] = -5; a[11][21] = -3;
  a[12][0] = 5; a[12][1] = 3; a[12][2] = -6; a[12][3] = 2; a[12][4] = 6; a[12][5] = -3; a[12][6] = 2; a[12][7] = 4; a[12][8] = 6; a[12][9] = 6; a[12][10] = 8; a[12][11] = -5; a[12][12] = -7; a[12][13] = -5; a[12][14] = -4; a[12][15] = 1; a[12][16] = -2; a[12][17] = -8; a[12][18] = -1; a[12][19] = -2; a[12][20] = 6; a[12][21] = 0;
  a[13][0] = 1; a[13][1] = -4; a[13][2] = -1; a[13][3] = -1; a[13][4] = -4; a[13][5] = -8; a[13][6] = 8; a[13][7] = 3; a[13][8] = -4; a[13][9] = 2; a[13][10] = 7; a[13][11] = 2; a[13][12] = 8; a[13][13] = 6; a[13][14] = -1; a[13][15] = 4; a[13][16] = -7; a[13][17] = -4; a[13][18] = 6; a[13][19] = 5; a[13][20] = 4; a[13][21] = -5;
  a[14][0] = 2; a[14][1] = -2; a[14][2] = 3; a[14][3] = 4; a[14][4] = -2; a[14][5] = -7; a[14][6] = 6; a[14][7] = -2; a[14][8] = 2; a[14][9] = -4; a[14][10] = -2; a[14][11] = 0; a[14][12] = -4; a[14][13] = 3; a[14][14] = 4; a[14][15] = -1; a[14][16] = 0; a[14][17] = 3; a[14][18] = -8; a[14][19] = -2; a[14][20] = 8; a[14][21] = 1;
  a[15][0] = 4; a[15][1] = 6; a[15][2] = 2; a[15][3] = -8; a[15][4] = -3; a[15][5] = -3; a[15][6] = -7; a[15][7] = -3; a[15][8] = -1; a[15][9] = 3; a[15][10] = -2; a[15][11] = 4; a[15][12] = 5; a[15][13] = 5; a[15][14] = 5; a[15][15] = -3; a[15][16] = 6; a[15][17] = -2; a[15][18] = 0; a[15][19] = -7; a[15][20] = -1; a[15][21] = 1;
  a[16][0] = -5; a[16][1] = 1; a[16][2] = 6; a[16][3] = -8; a[16][4] = -1; a[16][5] = -4; a[16][6] = -8; a[16][7] = 2; a[16][8] = -8; a[16][9] = 3; a[16][10] = -6; a[16][11] = -2; a[16][12] = 3; a[16][13] = 3; a[16][14] = 4; a[16][15] = -5; a[16][16] = -3; a[16][17] = -2; a[16][18] = -8; a[16][19] = 7; a[16][20] = 1; a[16][21] = 2;
  a[17][0] = 4; a[17][1] = 7; a[17][2] = -3; a[17][3] = 6; a[17][4] = 4; a[17][5] = 4; a[17][6] = -8; a[17][7] = -4; a[17][8] = -8; a[17][9] = -5; a[17][10] = 4; a[17][11] = -2; a[17][12] = -3; a[17][13] = 5; a[17][14] = -5; a[17][15] = -4; a[17][16] = -7; a[17][17] = -4; a[17][18] = 0; a[17][19] = -8; a[17][20] = 0; a[17][21] = -8;
  a[18][0] = -3; a[18][1] = -2; a[18][2] = -7; a[18][3] = 8; a[18][4] = 0; a[18][5] = 6; a[18][6] = -8; a[18][7] = 7; a[18][8] = -4; a[18][9] = -4; a[18][10] = -8; a[18][11] = -5; a[18][12] = -4; a[18][13] = -6; a[18][14] = 1; a[18][15] = 7; a[18][16] = 4; a[18][17] = -6; a[18][18] = -1; a[18][19] = 0; a[18][20] = 5; a[18][21] = 0;
  a[19][0] = -4; a[19][1] = -1; a[19][2] = 0; a[19][3] = 1; a[19][4] = -4; a[19][5] = -4; a[19][6] = -8; a[19][7] = 7; a[19][8] = 3; a[19][9] = 0; a[19][10] = 5; a[19][11] = 5; a[19][12] = -8; a[19][13] = 5; a[19][14] = 1; a[19][15] = 0; a[19][16] = 3; a[19][17] = -8; a[19][18] = 8; a[19][19] = 0; a[19][20] = -8; a[19][21] = -1;
  a[20][0] = -7; a[20][1] = 6; a[20][2] = 0; a[20][3] = -2; a[20][4] = -4; a[20][5] = -5; a[20][6] = 0; a[20][7] = 5; a[20][8] = -3; a[20][9] = -5; a[20][10] = 0; a[20][11] = -3; a[20][12] = 0; a[20][13] = 3; a[20][14] = 1; a[20][15] = 8; a[20][16] = -8; a[20][17] = 7; a[20][18] = 1; a[20][19] = 8; a[20][20] = -8; a[20][21] = -5;
  a[21][0] = 2; a[21][1] = -7; a[21][2] = -5; a[21][3] = -4; a[21][4] = 5; a[21][5] = -7; a[21][6] = -8; a[21][7] = 6; a[21][8] = -5; a[21][9] = -6; a[21][10] = -7; a[21][11] = -5; a[21][12] = -8; a[21][13] = 5; a[21][14] = 5; a[21][15] = -3; a[21][16] = 7; a[21][17] = -6; a[21][18] = 2; a[21][19] = -7; a[21][20] = -2; a[21][21] = -3;
  a[22][0] = -6; a[22][1] = 2; a[22][2] = -1; a[22][3] = 1; a[22][4] = 0; a[22][5] = 8; a[22][6] = -1; a[22][7] = 6; a[22][8] = 8; a[22][9] = 7; a[22][10] = -7; a[22][11] = 2; a[22][12] = 2; a[22][13] = -7; a[22][14] = 6; a[22][15] = -4; a[22][16] = 6; a[22][17] = -2; a[22][18] = -6; a[22][19] = 3; a[22][20] = -2; a[22][21] = -8;

  b[0][0] = 1; b[0][1] = -5; b[0][2] = -2; b[0][3] = 5; b[0][4] = -3; b[0][5] = -6; b[0][6] = -4; b[0][7] = -3; b[0][8] = 7; b[0][9] = -6; b[0][10] = -7; b[0][11] = -1; b[0][12] = 3; b[0][13] = -3; b[0][14] = 2; b[0][15] = -6; b[0][16] = 1; b[0][17] = -8; b[0][18] = 4; b[0][19] = -6; b[0][20] = 7;
  b[1][0] = -1; b[1][1] = 6; b[1][2] = 4; b[1][3] = 7; b[1][4] = 2; b[1][5] = 2; b[1][6] = 1; b[1][7] = -2; b[1][8] = 6; b[1][9] = -4; b[1][10] = 1; b[1][11] = -5; b[1][12] = 6; b[1][13] = -1; b[1][14] = 7; b[1][15] = 6; b[1][16] = 3; b[1][17] = -4; b[1][18] = 6; b[1][19] = -5; b[1][20] = -3;
  b[2][0] = 1; b[2][1] = 7; b[2][2] = 8; b[2][3] = 0; b[2][4] = 7; b[2][5] = 2; b[2][6] = -4; b[2][7] = 7; b[2][8] = 8; b[2][9] = 7; b[2][10] = 8; b[2][11] = -6; b[2][12] = 3; b[2][13] = 0; b[2][14] = 8; b[2][15] = -2; b[2][16] = -6; b[2][17] = -3; b[2][18] = 5; b[2][19] = 6; b[2][20] = -3;
  b[3][0] = -3; b[3][1] = -2; b[3][2] = 0; b[3][3] = 0; b[3][4] = 0; b[3][5] = -3; b[3][6] = 6; b[3][7] = 8; b[3][8] = -7; b[3][9] = -8; b[3][10] = -4; b[3][11] = 0; b[3][12] = -3; b[3][13] = 8; b[3][14] = -2; b[3][15] = 3; b[3][16] = 2; b[3][17] = -8; b[3][18] = -5; b[3][19] = -7; b[3][20] = -5;
  b[4][0] = 7; b[4][1] = 4; b[4][2] = 8; b[4][3] = 3; b[4][4] = -3; b[4][5] = -3; b[4][6] = -2; b[4][7] = 8; b[4][8] = -7; b[4][9] = 1; b[4][10] = -4; b[4][11] = -4; b[4][12] = -3; b[4][13] = -3; b[4][14] = -2; b[4][15] = -2; b[4][16] = -2; b[4][17] = 4; b[4][18] = -5; b[4][19] = 8; b[4][20] = 0;
  b[5][0] = -5; b[5][1] = -2; b[5][2] = 8; b[5][3] = 4; b[5][4] = 5; b[5][5] = -7; b[5][6] = 7; b[5][7] = 6; b[5][8] = 8; b[5][9] = 1; b[5][10] = 4; b[5][11] = -2; b[5][12] = 0; b[5][13] = -1; b[5][14] = -4; b[5][15] = 3; b[5][16] = -8; b[5][17] = -1; b[5][18] = -4; b[5][19] = -4; b[5][20] = 4;
  b[6][0] = -1; b[6][1] = -1; b[6][2] = 7; b[6][3] = 0; b[6][4] = -3; b[6][5] = -7; b[6][6] = 5; b[6][7] = -1; b[6][8] = -8; b[6][9] = -7; b[6][10] = 2; b[6][11] = -8; b[6][12] = -7; b[6][13] = 0; b[6][14] = -6; b[6][15] = 8; b[6][16] = -2; b[6][17] = -8; b[6][18] = -4; b[6][19] = -3; b[6][20] = 5;
  b[7][0] = 6; b[7][1] = -1; b[7][2] = -2; b[7][3] = 8; b[7][4] = 4; b[7][5] = -8; b[7][6] = 5; b[7][7] = -3; b[7][8] = -4; b[7][9] = 4; b[7][10] = 7; b[7][11] = -1; b[7][12] = 3; b[7][13] = 8; b[7][14] = -4; b[7][15] = -6; b[7][16] = -6; b[7][17] = 5; b[7][18] = -8; b[7][19] = 4; b[7][20] = 2;
  b[8][0] = -5; b[8][1] = 3; b[8][2] = 5; b[8][3] = 5; b[8][4] = 7; b[8][5] = 5; b[8][6] = 5; b[8][7] = 1; b[8][8] = 8; b[8][9] = 4; b[8][10] = 1; b[8][11] = 1; b[8][12] = 1; b[8][13] = 8; b[8][14] = 7; b[8][15] = -4; b[8][16] = -2; b[8][17] = -4; b[8][18] = 2; b[8][19] = -2; b[8][20] = -1;
  b[9][0] = -6; b[9][1] = 3; b[9][2] = 6; b[9][3] = -5; b[9][4] = 1; b[9][5] = 4; b[9][6] = -5; b[9][7] = 7; b[9][8] = 0; b[9][9] = 6; b[9][10] = 8; b[9][11] = -5; b[9][12] = 7; b[9][13] = 5; b[9][14] = 8; b[9][15] = -4; b[9][16] = -7; b[9][17] = 3; b[9][18] = 5; b[9][19] = -3; b[9][20] = 8;
  b[10][0] = 3; b[10][1] = 3; b[10][2] = 7; b[10][3] = 3; b[10][4] = -3; b[10][5] = -4; b[10][6] = -1; b[10][7] = -1; b[10][8] = 7; b[10][9] = -3; b[10][10] = -5; b[10][11] = 4; b[10][12] = 7; b[10][13] = -7; b[10][14] = -3; b[10][15] = 3; b[10][16] = -8; b[10][17] = 5; b[10][18] = -2; b[10][19] = -1; b[10][20] = 7;
  b[11][0] = 1; b[11][1] = 5; b[11][2] = -3; b[11][3] = -5; b[11][4] = 7; b[11][5] = 2; b[11][6] = 4; b[11][7] = 7; b[11][8] = 7; b[11][9] = -7; b[11][10] = 2; b[11][11] = 7; b[11][12] = 0; b[11][13] = 4; b[11][14] = -8; b[11][15] = -7; b[11][16] = -7; b[11][17] = -8; b[11][18] = 3; b[11][19] = 8; b[11][20] = -3;
  b[12][0] = 3; b[12][1] = -5; b[12][2] = 6; b[12][3] = -4; b[12][4] = -4; b[12][5] = -8; b[12][6] = -3; b[12][7] = 8; b[12][8] = -4; b[12][9] = -1; b[12][10] = -8; b[12][11] = 8; b[12][12] = 2; b[12][13] = 4; b[12][14] = -6; b[12][15] = 8; b[12][16] = -2; b[12][17] = -5; b[12][18] = 1; b[12][19] = 3; b[12][20] = -6;
  b[13][0] = 4; b[13][1] = -4; b[13][2] = 6; b[13][3] = -7; b[13][4] = 3; b[13][5] = 8; b[13][6] = 7; b[13][7] = 8; b[13][8] = -3; b[13][9] = 3; b[13][10] = -8; b[13][11] = -4; b[13][12] = 4; b[13][13] = 8; b[13][14] = 6; b[13][15] = 3; b[13][16] = 1; b[13][17] = 5; b[13][18] = 1; b[13][19] = 3; b[13][20] = 8;
  b[14][0] = -5; b[14][1] = 3; b[14][2] = -7; b[14][3] = 7; b[14][4] = 6; b[14][5] = 8; b[14][6] = -2; b[14][7] = -6; b[14][8] = -8; b[14][9] = -2; b[14][10] = -4; b[14][11] = 7; b[14][12] = 3; b[14][13] = -8; b[14][14] = -7; b[14][15] = -1; b[14][16] = -1; b[14][17] = 0; b[14][18] = -2; b[14][19] = 0; b[14][20] = -7;
  b[15][0] = 2; b[15][1] = 8; b[15][2] = -8; b[15][3] = -1; b[15][4] = -3; b[15][5] = 1; b[15][6] = 7; b[15][7] = 4; b[15][8] = -4; b[15][9] = 2; b[15][10] = 2; b[15][11] = -1; b[15][12] = -4; b[15][13] = 4; b[15][14] = -1; b[15][15] = -8; b[15][16] = 4; b[15][17] = 3; b[15][18] = 4; b[15][19] = -7; b[15][20] = -3;
  b[16][0] = -7; b[16][1] = -3; b[16][2] = -2; b[16][3] = 7; b[16][4] = 7; b[16][5] = 5; b[16][6] = -6; b[16][7] = 0; b[16][8] = 3; b[16][9] = -3; b[16][10] = 7; b[16][11] = 7; b[16][12] = -5; b[16][13] = 4; b[16][14] = -5; b[16][15] = 0; b[16][16] = 0; b[16][17] = -1; b[16][18] = -5; b[16][19] = 5; b[16][20] = 3;
  b[17][0] = -5; b[17][1] = -3; b[17][2] = 0; b[17][3] = -6; b[17][4] = -8; b[17][5] = 3; b[17][6] = 3; b[17][7] = 3; b[17][8] = 5; b[17][9] = -5; b[17][10] = -4; b[17][11] = 0; b[17][12] = -5; b[17][13] = -7; b[17][14] = -3; b[17][15] = -2; b[17][16] = -3; b[17][17] = 6; b[17][18] = -8; b[17][19] = -3; b[17][20] = 4;
  b[18][0] = -4; b[18][1] = 8; b[18][2] = 2; b[18][3] = -6; b[18][4] = -5; b[18][5] = 3; b[18][6] = -2; b[18][7] = 5; b[18][8] = 1; b[18][9] = 6; b[18][10] = -1; b[18][11] = -3; b[18][12] = -1; b[18][13] = 7; b[18][14] = 1; b[18][15] = 6; b[18][16] = 8; b[18][17] = -7; b[18][18] = -6; b[18][19] = 1; b[18][20] = 5;
  b[19][0] = 5; b[19][1] = -8; b[19][2] = 8; b[19][3] = 4; b[19][4] = 3; b[19][5] = -1; b[19][6] = -5; b[19][7] = 0; b[19][8] = 3; b[19][9] = 4; b[19][10] = -4; b[19][11] = 3; b[19][12] = 6; b[19][13] = -5; b[19][14] = 0; b[19][15] = -5; b[19][16] = -8; b[19][17] = -7; b[19][18] = 4; b[19][19] = -6; b[19][20] = -4;
  b[20][0] = -7; b[20][1] = 3; b[20][2] = -5; b[20][3] = 0; b[20][4] = -5; b[20][5] = -4; b[20][6] = -2; b[20][7] = 1; b[20][8] = -3; b[20][9] = -3; b[20][10] = -1; b[20][11] = 5; b[20][12] = -3; b[20][13] = 2; b[20][14] = -8; b[20][15] = -1; b[20][16] = -3; b[20][17] = -3; b[20][18] = 3; b[20][19] = 1; b[20][20] = 0;
  b[21][0] = 1; b[21][1] = -5; b[21][2] = 6; b[21][3] = 0; b[21][4] = -2; b[21][5] = 2; b[21][6] = -3; b[21][7] = 2; b[21][8] = -3; b[21][9] = 7; b[21][10] = 5; b[21][11] = 8; b[21][12] = 0; b[21][13] = 0; b[21][14] = 0; b[21][15] = -6; b[21][16] = 5; b[21][17] = -7; b[21][18] = 2; b[21][19] = 6; b[21][20] = 6;

  let _ = matmul(23,22,21,a,b,c);
  let _ = matshow(23, 22, a);
  let _ = print_endline();
  let _ = matshow(22, 21, b);
  let _ = print_endline();
  let _ = matshow(23, 21, c);
  ()

};