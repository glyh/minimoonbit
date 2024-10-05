typealias S = @types.Syntax

typealias T = @types.Type

typealias N = @types.Name

enum Either[L, R] {
  Left(L)
  Right(R)
} derive(Hash, Eq)
