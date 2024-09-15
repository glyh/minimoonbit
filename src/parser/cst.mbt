struct Ann {
  src_start : @lex.Loc
  src_end : @lex.Loc?
}

type Program Array[Top]

type! ParseError {
  Unexpected((@lex.Token, @lex.Loc, @lex.Loc))
  EarlyEOF
  Missing(@lex.Token, @lex.Loc)
  ExpectButEOF(@lex.Token)
  Unreachable
  TopLevelNotColNot1((@lex.Token, @lex.Loc, @lex.Loc))
} derive(Show)

enum TopNaked {
  LetDecl(ID, Type, Exp)
  FnDecl(ID, Result[Array[(ID, Type)], ParseError], Type, Stmt)
  MainDecl(Stmt)
  InitDecl(Stmt)
}

type Top Result[(TopNaked, Ann), ParseError]

type ID Result[(String, Ann), ParseError]

enum StmtHeadNaked {
  LetTuple(Array[ID], Type?, Exp)
  Let(ID, Type?, Exp)
  NontopFn(ID, Array[(ID, Type?)], Type?, Stmt)
  Assign(Exp, Exp)
}

type StmtHead Result[(StmtHeadNaked, Ann), ParseError]

type Stmt Result[(Array[StmtHead], Exp, Ann), ParseError]

enum Value {
  I32(Int)
  F64(Double)
  Bool(Bool)
  Unit
}

enum Unop {
  Not
  Neg
}

enum BinopTag {
  Eq
  Le
  Add
  Sub
  Mul
  Div
}

type Binop Result[(BinopTag, Ann), ParseError]

enum ExpNaked {
  Atom(Value)
  Tuple(Array[Exp])
  Block(Stmt)
  Unary(Unop, Exp)
  Binary(Binop, Exp, Exp)
  Apply(Exp, Array[Exp]) // we need at least 2 expression to perform an apply
  If(Exp, Stmt, Stmt?)
  Index(Exp, Exp)
  Var(ID)
  MakeArray(Exp, Exp) // (len, ele)
}

type Exp Result[(ExpNaked, Ann), ParseError]

enum TypeNaked {
  TUnit
  TBool
  TInt
  TDouble
  TArray(Type)
  TTuple(Array[Type])
  TFunction(Array[Type], Type)
}

type Type Result[(TypeNaked, Ann), ParseError]
