fn no_ann() -> Unit {
  fn main() { () };
  fn a(u: Int, v: Int, w) -> Int {
    fn init() { () };
    fn b(x: Bool, y, z: Bool) {
      ()
    }; 
    fn c(x: Bool, y, z: Bool) {
      ()
    }; fn g() { () };
    fn d(x: Bool, y: Int, z: Int) -> ((Unit) -> Int) -> Int {
      fn d(x: Bool, y: Int, z: Int) {
        fn d() -> Unit {
          ()
        }; ()
      }; ();
      fn ret(q) {
        q()
      };
      ret
    }; ()
  }; ()
};

fn main { () };
fn init { () };

fn wat () -> Int {()};
