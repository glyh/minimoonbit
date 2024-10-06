## Call Convention
1. We use `ra` to store the continuation address, as it's otherwise unused in our language. We do need to push it onto stack to preserve it's value when doing a native call, though.
2. We store `closure` pointer after any arguments, so we should be able to work with native functions just fine. This differs from what is being done in the book "Compiling with Continuations".
