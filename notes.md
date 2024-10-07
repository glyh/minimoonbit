## Call Convention
1. We use `ra` to store the continuation address, as it's otherwise unused in our language. We do need to push it onto stack to preserve it's value when doing a native call, though.
2. We store `closure` pointer after any arguments, so we should be able to work with native functions just fine. This differs from what is being done in the book "Compiling with Continuations".

## TODO
1. fix call convention for external functions. For example:
```
:print_int([?26, kont_main.4.22])
```
Inside print_int, we should do something like this:
```
fn_ptr.99 = kont_main.4.22.0
fn_ptr.99([result, kont_main.4.22])
```
