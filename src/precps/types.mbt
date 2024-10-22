typealias S = @types.Syntax

typealias T = @types.Type

enum Either[L, R] {
  Left(L)
  Right(R)
} derive(Hash, Eq)
