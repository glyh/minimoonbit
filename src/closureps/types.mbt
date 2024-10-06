typealias T = @top.Type

typealias PrimOp = @precps.PrimOp

typealias S = @cps.Cps

typealias Var = @cps.Var

typealias Value = @cps.Value

enum Either[L, R] {
  Left(L)
  Right(R)
}
