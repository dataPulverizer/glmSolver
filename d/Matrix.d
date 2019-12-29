import std.conv: to;
import std.traits: isFloatingPoint, isIntegral, isNumeric;
import std.algorithm: min, max, map, fold;
import std.stdio: File, write, writef, writeln, writefln;
import std.file: remove;
import std.format: format;
import std.math: atan, exp, expm1, log, modf, fabs, fmax, fmin, sqrt, cos, tan, PI;
import std.mathspecial : normalDistribution, normalDistributionInverse;
import std.random;
import std.typecons: Tuple, tuple;

alias fmin min;
alias fmax max;
alias PI pi;
alias normalDistribution pnorm;
alias normalDistributionInverse qnorm;
T dnorm(T)(T x)
{
  return (1/((2*pi)^^0.5)) * exp(-(x^^2)/2);
}

/* dmd Matrix.d -L-lopenblas -L-lpthread -L-llapacke -L-llapack -L-lm && ./Matrix */
/********************************************* Matrix CBLAS ENUMS *********************************************/
enum CBLAS_LAYOUT {
  CblasRowMajor = 101,
  CblasColMajor = 102
}

alias CBLAS_LAYOUT.CblasRowMajor CblasRowMajor;
alias CBLAS_LAYOUT.CblasColMajor CblasColMajor;

enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113
}

alias CBLAS_TRANSPOSE.CblasNoTrans CblasNoTrans;
alias CBLAS_TRANSPOSE.CblasTrans CblasTrans;
alias CBLAS_TRANSPOSE.CblasConjTrans CblasConjTrans;

// For Matrices
enum CBLAS_SYMMETRY {
  CblasGeneral          = 231,
  CblasSymmetric        = 232,
  CblasHermitian        = 233,
  CblasTriangular       = 234,
  CblasLowerTriangular  = 235,
  CblasUpperTriangular  = 236,
  CblasLowerSymmetric   = 237,
  CblasUpperSymmetric   = 238,
  CblasLowerHermitian   = 239,
  CblasUpperHermitian   = 240
};

alias CBLAS_SYMMETRY.CblasGeneral          CblasGeneral         ;
alias CBLAS_SYMMETRY.CblasSymmetric        CblasSymmetric       ;
alias CBLAS_SYMMETRY.CblasHermitian        CblasHermitian       ;
alias CBLAS_SYMMETRY.CblasTriangular       CblasTriangular      ;
alias CBLAS_SYMMETRY.CblasLowerTriangular  CblasLowerTriangular ;
alias CBLAS_SYMMETRY.CblasUpperTriangular  CblasUpperTriangular ;
alias CBLAS_SYMMETRY.CblasLowerSymmetric   CblasLowerSymmetric  ;
alias CBLAS_SYMMETRY.CblasUpperSymmetric   CblasUpperSymmetric  ;
alias CBLAS_SYMMETRY.CblasLowerHermitian   CblasLowerHermitian  ;
alias CBLAS_SYMMETRY.CblasUpperHermitian   CblasUpperHermitian  ;
/********************************************* Printer Utility Functions *********************************************/
int getMaxLength(T)(const(T[]) v)
if(isIntegral!T)
{
  int x = 0;
  foreach(el; v)
    x = max(x, cast(int)to!string(el).length);
  return x;
}

int[] getMaxLength(T)(const(T[]) v)
if(isFloatingPoint!T)
{
  int[] x = [0, 0];
  real intpart, frac;
  foreach(el; v)
  {
    frac = modf(cast(real)el, intpart);
    x[0] = max(x[0], cast(int)to!string(intpart).length);
    x[1] = max(x[1], cast(int)to!string(frac).length);
  }
  return x;
}
string oldFloatFormat(int[] mlen, int dig = 5, int gap = 2)
{
  int tot = mlen[0] + mlen[1];
  if(tot > dig)
  {
    if(mlen[0] > 4)
    {
      mlen[0] = min(mlen[0], 5);
      mlen[1] = 2;
    } else {
      mlen[0] = 2;
      mlen[1] = min(mlen[1], 5);
    }
  }
  string dform = "%" ~ to!string(mlen[0] + mlen[1] + gap) ~ "." ~ to!string(mlen[1]) ~ "f";
  return dform;
}

string floatFormat(int[] mlen, int dp = 6, int dig = 7, int gap = 2)
{
  string dform = "";
  int tot = mlen[0] + mlen[1];

  if((tot > dig) && (mlen[0] > 1))
  {
    dform = "%" ~ to!string(dp + 4*gap) ~ "." ~ to!string(dp) ~ "e";
  } else if(tot > dig){
    dform = "%" ~ to!string(dp + 2*gap) ~ "." ~ to!string(dp) ~ "f";
  } else {
    dform = "%" ~ to!string(mlen[0] + mlen[1] + gap) ~ "." ~ to!string(mlen[1]) ~ "f";
  }

  return dform;
}
/********************************************* Matrix Class *********************************************/
mixin template MatrixGubbings(T, CBLAS_LAYOUT L)
{
  private:
    T[] data;
    ulong[] dim;
  public:
    this(T)(T[] dat, ulong rows, ulong cols)
    {
      assert(rows*cols == dat.length, 
            "dimension of matrix inconsistent with length of array");
      data = dat; dim = [rows, cols];
    }
    this(ulong n, ulong m)
    {
      data = new T[n*m];
      dim = [n, m];
    }
    this(T)(T[] dat, ulong[] d)
    {
      ulong tlen = d[0]*d[1];
      assert(tlen == dat.length, 
            "dimension of matrix inconsistent with length of array");
      data = dat; dim = d;
    }
    this(T, CBLAS_LAYOUT L)(Matrix!(T, L) mat)
    {
      data = mat.data.dup;
      dim = mat.dim.dup;
    }
    @property Matrix!(T, L) dup()
    {
      return new Matrix!(T, L)(data.dup, dim.dup);
    }
    T opIndex(ulong i, ulong j) const
    {
      return data[dim[0]*j + i];
    }
    void opIndexAssign(T x, ulong i, ulong j)
    {
      // Zero based indexing
      // Need check for j < n_col & i < n_row
      data[dim[0]*j + i] = x;
    }
    void opIndexOpAssign(string op)(T x, ulong i, ulong j)
    {
      static if((op == "+") | (op == "-") | (op == "*") | (op == "/"))
        mixin("return data[dim[0]*j + i] " ~ op ~ "= x;");
      else static assert(0, "Operator "~ op ~" not implemented");
    }
    @property ulong nrow()
    {
      return dim[0];
    }
    @property ulong ncol()
    {
      return dim[1];
    }
    @property T[] getData()
    {
      return data;
    }
    @property ulong len()
    {
      return data.length;
    }
    /* Returns transposed matrix no duplication */
    Matrix!(T, L) t()
    {
      auto ddat = data.dup;
      ulong[] ddim = new ulong[2];
      ddim[0] = dim[1]; ddim[1] = dim[0];
      if((dim[0] == 1) & (dim[1] == 1)){
      } else if(dim[0] != dim[1]) {
        for(int j = 0; j < dim[1]; ++j)
        {
          for(int i = 0; i < dim[0]; ++i)
          {
            ddat[ddim[0]*i + j] = data[dim[0]*j + i];
          }
        }
      } else  if(dim[0] == dim[1]) {
        for(int j = 0; j < dim[1]; ++j)
        {
          for(int i = 0; i < dim[0]; ++i)
          {
            if(i == j)
              continue;
            ddat[ddim[0]*i + j] = data[dim[0]*j + i];
          }
        }
      }
      return new Matrix!(T, L)(ddat, ddim);
    }
    /* Cast to Column Vector */
    ColumnVector!(T) opCast(V: ColumnVector!(T))() {
      assert(ncol == 1, "The number of columns in the matrix 
           is not == 1 and so can not be converted to a matrix.");
      return new ColumnVector!(T)(data);
    }
    /* Cast to Row Vector */
    RowVector!(T) opCast(V: RowVector!(T))() {
      assert(ncol == 1, "The number of columns in the matrix
           is not == 1 and so can not be converted to a matrix.");
      return new RowVector!(T)(data);
    }
}
/* Assuming column major */
class Matrix(T, CBLAS_LAYOUT L = CblasColMajor)
if(isFloatingPoint!T)
{
  mixin MatrixGubbings!(T, L);
  override string toString() const
  {
    int[] mlen = getMaxLength!T(data);
    string dform = floatFormat(mlen);
    writeln(dform);
    string repr = format(" Matrix(%d x %d)\n", dim[0], dim[1]);
    for(int i = 0; i < dim[0]; ++i)
    {
      for(int j = 0; j < dim[1]; ++j)
      {
        repr ~= format(dform, opIndex(i, j));
      }
      repr ~= "\n";
    }
    //repr ~= "\n";
    return repr;
  }
}

class Matrix(T, CBLAS_LAYOUT L = CblasColMajor)
if(isIntegral!T)
{
  mixin MatrixGubbings!(T, L);
  override string toString() const
  {
    int dig = 6;
    int mlen = getMaxLength!T(data);
    int gap = 2;
    dig = mlen < dig ? mlen : dig;
    string dform = "%" ~ to!string(dig + gap) ~ "d";
    string repr = format(" Matrix(%d x %d)\n", dim[0], dim[1]);
    for(int i = 0; i < dim[0]; ++i)
    {
      for(int j = 0; j < dim[1]; ++j)
      {
        repr ~= format(dform, opIndex(i, j));
      }
      repr ~= "\n";
    }
    //repr ~= "\n";
    return repr;
  }
}

/* Convinient function for constructor with type inference */
Matrix!(T, L) matrix(T, CBLAS_LAYOUT L = CblasColMajor)(T[] dat, ulong rows, ulong cols)
{
  assert(rows*cols == dat.length, 
        "dimension of matrix inconsistent with length of array");
  return new Matrix!(T, L)(dat, rows, cols);
}
/* Constructor for matrix with data and dimension array */
Matrix!(T, L) matrix(T, CBLAS_LAYOUT L = CblasColMajor)(T[] data, ulong[] dim)
{
  assert(dim[0]*dim[1] == data.length, 
        "dimension of matrix inconsistent with length of array");
  return new Matrix!(T, L)(data, dim);
}
Matrix!(T, L) matrix(T, CBLAS_LAYOUT L = CblasColMajor)(T[] dat, ulong rows)
{
  assert(rows * rows == dat.length, 
        "dimension of matrix inconsistent with length of array");
  return new Matrix!(T, L)(dat, [rows, rows]);
}
/* Constructor for square matrix */
Matrix!(T, L) matrix(T, CBLAS_LAYOUT L = CblasColMajor)(Matrix!(T, L) m)
{
  return new Matrix!(T, L)(m);
}
/* Product of an array of elements */
T prod(T)(T[] x)
{
  ulong n = x.length;
  T ret = 1;
  for(ulong i = 0; i < n; ++i)
    ret *= x[i];
  return ret;
}
/* Create a matrix with shape dim where each element is x */
Matrix!(T, layout) fillMatrix(T, CBLAS_LAYOUT layout = CblasColMajor)(T x, ulong[] dim)
{
  ulong n = dim[0] * dim[1];
  T [] data = new T[n];
  for(ulong i = 0; i < n; ++i)
  {
    data[i] = x;
  }
  return new Matrix!(T, layout)(data, dim);
}
/********************************************* Vector Classes *********************************************/
interface Vector(T)
{
  @property ulong len() const;
  @property ulong length() const;
  T opIndex(ulong i) const;
  void opIndexAssign(T x, ulong i);
  void opIndexOpAssign(string op)(T x, ulong i);
  @property T[] getData();
  Matrix!(T, layout) opCast(M: Matrix!(T, layout), CBLAS_LAYOUT layout)();
}
mixin template VectorGubbings(T)
{
  T[] data;
  @property ulong len() const
  {
    return /* cast(ulong) */ data.length;
  }
  @property ulong length() const
  {
    return data.length;
  }
  this(T)(T[] dat)
  {
    data = dat;
  }
  this(ulong n)
  {
    data = new T[n];
  }
  T opIndex(ulong i) const
  {
    return data[i];
  }
  void opIndexAssign(T x, ulong i)
  {
    data[i] = x;
  }
  T opIndexOpAssign(string op)(T x, ulong i)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/"))
      mixin("return data[i] " ~ op ~ "= x;");
    else static assert(0, "Operator "~ op ~" not implemented");
  }
  @property T[] getData()
  {
    return data;
  }
  Matrix!(T, layout) opCast(M: Matrix!(T, layout), CBLAS_LAYOUT layout)()
  {
    return new Matrix!(T, layout)(data, [len, 1]);
  }
}
class ColumnVector(T) : Vector!T
if(isNumeric!T)
{
  mixin VectorGubbings!(T);

  override string toString() const
  {
    auto n = len();
    string repr = format("ColumnVector(%d)", n) ~ "\n";
    for(int i = 0; i < n; ++i)
    {
      repr ~= to!string(data[i]) ~ "\n";
    }
    return repr;
  }
  ColumnVector!T opBinary(string op)(ColumnVector!T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/"))
    {
      assert(data.length == rhs.data.length, "Vector lengths are not the same.");
      auto ret = new ColumnVector!T(rhs.data.dup);
      for(ulong i = 0; i < data.length; ++i)
        mixin("ret.data[i] = ret.data[i] " ~ op ~ " data[i];");
      return ret;
    } else static assert(0, "Operator "~ op ~" not implemented");
  }
  void opOpAssign(string op)(ColumnVector!T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/"))
    {
      assert(data.length == rhs.data.length, "Vector lengths are not the same.");
      for(ulong i = 0; i < data.length; ++i)
        mixin("data[i] = data[i] " ~ op ~ " rhs.data[i];");
      return;
    } else static assert(0, "Operator "~ op ~" not implemented");
  }
  @property ColumnVector!(T) dup()
  {
    return new ColumnVector!T(data.dup);
  }
}
class RowVector(T): Vector!T
if(isNumeric!T)
{
  mixin VectorGubbings!(T);

  override string toString() const
  {
    int dig = 5;
    string repr = format("RowVector(%d)", len()) ~ "\n" ~ to!string(data) ~ "\n";
    return repr;
  }
  RowVector!T opBinary(string op)(RowVector!T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/"))
    {
      assert(data.len == rhs.data.len, "Vector lengths are not the same.");
      auto ret = RowVector!T(rhs.data.dup);
      for(ulong i = 0; i < data.len; ++i)
        mixin("ret.data[i] = ret.data[i] "~ op ~ " data[i]");
      return ret;
    } else static assert(0, "Operator "~ op ~" not implemented");
  }
  void opOpAssign(string op)(RowVector!T rhs)
  {
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/"))
    {
      assert(data.len == rhs.data.len, "Vector lengths are not the same.");
      for(ulong i = 0; i < data.len; ++i)
        mixin("data[i] = data[i] "~ op ~ " rhs.data[i]");
      return;
    } else static assert(0, "Operator "~ op ~" not implemented");
  }
  @property RowVector!(T) dup()
  {
    return new RowVector!T(data.dup);
  }
}
/********************************************* Testing Functions *********************************************/
/* Testing allowing template parameters to be passed on using parameters */
void dispatchTemplate(T, CBLAS_LAYOUT L)(Matrix!(T, L) m)
{
  writeln("Matrix Layout: ", L, ", with element type: ", T.stringof, "\n");
}
/********************************************* Convenient Matrix Functions *********************************************/
Matrix!T createRandomMatrix(T = double)(ulong m)
{
  Mt19937_64 gen;
  gen.seed(unpredictableSeed);
  ulong len = m*m;
  T[] data = new T[len];
  for(int i = 0; i < len; ++i)
    data[i] = uniform01!(T)(gen);
  return new Matrix!T(data, [m, m]);
}
Matrix!T createRandomMatrix(T = double)(ulong m, ulong n)
{
  Mt19937_64 gen;
  gen.seed(unpredictableSeed);
  ulong len = m*n;
  T[] data = new T[len];
  for(int i = 0; i < len; ++i)
    data[i] = uniform01!(T)(gen);
  return new Matrix!T(data, [m, n]);
}
Matrix!(T, layout) createSymmetricMatrix(T, CBLAS_LAYOUT layout = CblasColMajor)(ulong m)
{
  ulong n = m + (m*m - m)/2;
  auto dat = createRandomArray!(T)(n);
  auto dim = [m, m];
  auto data = new T[m*m];
  /*
    dat is only as long symmetrix data for upper matrix
    row major.
  */
  int k = 0;
  for(int j = 0; j < m; ++j)
  {
    for(int i = 0; (i <= j) && (j < m); ++i)
    {
      data[dim[0]*j + i] = dat[k];
      ++k;
    }
  }
  for(int j = 0; j < m; ++j)
  {
    for(int i = j; i < m; ++i)
    {
      if(i == j)
        continue;
      data[dim[0]*j + i] = data[dim[0]*i + j];
    }
  }
  return new Matrix!(T, layout)(data, dim);
}
ColumnVector!(T) columnVector(T)(T[] data)
{
  return new ColumnVector!(T)(data);
}
/********************************************* Convenient Vector Functions *********************************************/
auto fillColumn(T)(T x, ulong n)
{
  auto v = new T[n];
  for(ulong i = 0; i < n; ++i)
    v[i] = x;
  return new ColumnVector!T(v);
}
auto zerosColumn(T)(ulong n)
{
  //auto v = new T[n];
  //for(ulong i = 0; i < n; ++i)
  //  v[i] = 0;
  //return new ColumnVector!T(v);
  return fillColumn!(T)(0, n);
}
auto onesColumn(T)(ulong n)
{
  return fillColumn!(T)(1, n);
}
RowVector!(T) rowVector(T)(T[] data)
{
  return new RowVector!(T)(data);
}
T sum(T)(Vector!T v)
if(isNumeric!T)
{
  return fold!((a, b) => a + b)(v.getData);
}
T[] createRandomArray(T = double)(ulong m)
{
  Mt19937_64 gen;
  gen.seed(unpredictableSeed);
  T[] data = new T[m];
  for(int i = 0; i < m; ++i)
    data[i] = uniform01!(T)(gen);
  return data;
}
ColumnVector!T createRandomColumnVector(T = double)(ulong m)
{
  Mt19937_64 gen;
  gen.seed(unpredictableSeed);
  T[] data = new T[m];
  for(int i = 0; i < m; ++i)
    data[i] = uniform01!(T)(gen);
  return new ColumnVector!T(data);
}
RowVector!T createRandomRowVector(T = double)(ulong m)
{
  Mt19937_64 gen;
  gen.seed(unpredictableSeed);
  T[] data = new T[m];
  for(int i = 0; i < m; ++i)
    data[i] = uniform01!(T)(gen);
  return new RowVector!T(data);
}
/********************************************* Sweep & Map *********************************************/
enum Axis{
  Row = 0,
  Column = 1
}
alias Axis.Row Row;
alias Axis.Column Column;
/*
** Sweep allows us to sweep a vector Column-wise Axis.Column or 
** Row-wise Axis.Row.
*/
Matrix!T sweep(alias fun, Axis index = Column, T)(Matrix!T m, Vector!T v)
{
  static if(index == Column)
    assert(m.nrow == v.len, "Number of rows differs from length of vector");
  else if(index == Row)
    assert(m.ncol == v.len, "Number of columns differs from length of vector");
  auto ret = new Matrix!T(m);
  static if(index == Column)
  {
    for(int j = 0; j < m.ncol; ++j)
    {
      for(int i = 0; i < m.nrow; ++i)
      {
        ret[i, j] = fun(ret[i, j], v[i]);
      }
    }
  }else
  {
    for(int j = 0; j < m.ncol; ++j)
    {
      for(int i = 0; i < m.nrow; ++i)
      {
        ret[i, j] = fun(ret[i, j], v[j]);
      }
    }
  }
  return ret;
}

/* Reference version for sweep(Matrix, Vector) */
void sweepRef(alias fun, Axis index = Column, T)(Matrix!T m, Vector!T v)
{
  static if(index == Column)
    assert(m.nrow == v.len, "Number of rows differs from length of vector");
  else if(index == Row)
    assert(m.ncol == v.len, "Number of columns differs from length of vector");
  static if(index == Column)
  {
    for(int j = 0; j < m.ncol; ++j)
    {
      for(int i = 0; i < m.nrow; ++i)
      {
        m[i, j] = fun(m[i, j], v[i]);
      }
    }
  }else
  {
    for(int j = 0; j < m.ncol; ++j)
    {
      for(int i = 0; i < m.nrow; ++i)
      {
        m[i, j] = fun(m[i, j], v[j]);
      }
    }
  }
  return;
}

/* Matrix-Array sweep function */
Matrix!T sweep(alias fun, Axis index = Column, T)(Matrix!T m, T[] v)
{
  static if(index == Column)
    assert(m.nrow == v.length, "Number of rows differs from length of vector");
  else if(index == Row)
    assert(m.ncol == v.length, "Number of columns differs from length of vector");
  
  auto ret = new Matrix!T(m);
  static if(index == Column)
  {
    for(int j = 0; j < m.ncol; ++j)
    {
      for(int i = 0; i < m.nrow; ++i)
      {
        ret[i, j] = fun(ret[i, j], v[i]);
      }
    }
  }else
  {
    for(int j = 0; j < m.ncol; ++j)
    {
      for(int i = 0; i < m.nrow; ++i)
      {
        ret[i, j] = fun(ret[i, j], v[j]);
      }
    }
  }
  return ret;
}

/* Matrix-Vector sweep function */
Matrix!T sweep(alias fun, T)(Matrix!T m, Matrix!T n)
{
  assert((m.nrow == n.nrow) & (m.ncol == n.ncol), "Number of rows or columns in matrix differs");
  auto ret = new Matrix!T(m);
  for(int j = 0; j < m.ncol; ++j)
  {
    for(int i = 0; i < m.nrow; ++i)
    {
      ret[i, j] = fun(ret[i, j], n[i, j]);
    }
  }
  return ret;
}

/* Vector-Vector map functions */
ColumnVector!T map(alias fun, T)(ColumnVector!T v1, ColumnVector!T v2)
{
  assert(v1.len == v2.len, "Length of vectors are unequal");
  auto ret = new ColumnVector!T(v1.len);
  for(int i = 0; i < ret.len; ++i)
    ret[i] = fun(v1[i], v2[i]);
  return ret;
}
RowVector!T map(alias fun, T)(RowVector!T v1, RowVector!T v2)
{
  assert(v1.len == v2.len, "Length of vectors are unequal");
  auto ret = new RowVector!T(v1.len);
  for(int i = 0; i < ret.len; ++i)
    ret[i] = fun(v1[i], v2[i]);
  return ret;
}
ColumnVector!T map(alias fun, T)(ColumnVector!T v1, ColumnVector!T v2, ColumnVector!T v3)
{
  assert((v1.len == v2.len) & (v1.len == v3.len), "Length of vectors are unequal");
  auto ret = new ColumnVector!T(v1.len);
  for(int i = 0; i < ret.len; ++i)
    ret[i] = fun(v1[i], v2[i], v3[i]);
  return ret;
}

/* Mapping function for Vectors */
T[] map(alias fun, T)(T[] v)
{
  T[] ret = new T[v.len];
  for(int i = 0; i < ret.len; ++i)
    ret[i] = fun(v[i]);
  return ret;
}
T[] map(alias fun, T)(T[] v1, T[] v2)
{
  assert(v1.length == v2.length, "The lengths of the two vectors are not equal");
  T[] ret = new T[v1.length];
  for(int i = 0; i < v1.length; ++i)
    ret[i] = fun(v1[i], v2[i]);
  return ret;
}

ColumnVector!T map(alias fun, T)(ColumnVector!T v)
{
  auto ret = new ColumnVector!T(v.len);
  for(int i = 0; i < ret.len; ++i)
    ret[i] = fun(v[i]);
  return ret;
}
RowVector!T map(alias fun, T)(RowVector!T v)
{
  auto ret = new RowVector!T(v.len);
  for(int i = 0; i < ret.len; ++i)
    ret[i] = fun(v[i]);
  return ret;
}
/********************************************* CBLAS & Lapack Imports *********************************************/
extern(C) @nogc nothrow{
  void cblas_dgemm(in CBLAS_LAYOUT layout, in CBLAS_TRANSPOSE TransA,
                   in CBLAS_TRANSPOSE TransB, in int M, in int N,
                   in int K, in double alpha, in double  *A,
                   in int lda, in double  *B, in int ldb,
                   in double beta, double  *C, in int ldc);
  void cblas_dgemv(in CBLAS_LAYOUT layout, in CBLAS_TRANSPOSE TransA, 
                   in int M, in int N, in double alpha, in double *A, 
                   in int lda, in double *X, in int incx, in double beta, 
                   double *Y, in int incy);
  
  /* See IBM ESSL documentation for more details */
  int LAPACKE_dgetrf(int matrix_layout, int m,
  	        int n, double* a, int lda, 
  	        int* ipiv);
  int LAPACKE_dgetri(int matrix_layout, int n, 
  	        double* a, int lda, in int* ipiv);
  int LAPACKE_dpotrf(int matrix_layout, char uplo, int n,
            double* a, int lda);
  int LAPACKE_dpotri(int matrix_layout, char uplo, int n, 
            double* a, int lda);

  int LAPACKE_dgetrs(int matrix_layout, char trans, int n , int nrhs, 
          in double* a, int lda , in int* ipiv, double* b, int ldb);
  int LAPACKE_dpotrs(int matrix_layout, char uplo, int n, int nrhs, 
          in double* a, int lda, double* b, int ldb);

  /* Norm of an array */
  double cblas_dnrm2(in int n , in double* x , in int incx);
  int LAPACKE_dgesvd(int matrix_layout, char jobu, char jobvt, int m, int n, double* a, 
                      int lda, double* s, double* u, int ldu, double* vt, int ldvt, double* superb);
  int LAPACKE_dgeqrf(int matrix_layout, int m, int n, double* a, int lda, double* tau);
  int LAPACKE_dtrtrs(int matrix_layout, char uplo, char trans, char diag, int n, int nrhs, in double* a, int lda , double* b, int ldb);
  int LAPACKE_dorgqr(int matrix_layout, int m, int n, int k, double* a, int lda, in double* tau);
  /* Set the number of threads for blas/lapack in openblas */
  void openblas_set_num_threads(int num_threads);
}

alias cblas_dgemm dgemm;
alias cblas_dgemv dgemv;
alias LAPACKE_dgetrf dgetrf;
alias LAPACKE_dgetri dgetri;
alias LAPACKE_dgesvd dgesvd;
alias LAPACKE_dpotrf dpotrf;
alias LAPACKE_dpotri dpotri;
alias LAPACKE_dgetrs dgetrs;
alias LAPACKE_dpotrs dpotrs;
alias LAPACKE_dgeqrf dgeqrf;
alias LAPACKE_dtrtrs dtrtrs;
alias LAPACKE_dorgqr dorgqr;

/* Norm function */
double norm(int incr = 1)(double[] x)
{
	return cblas_dnrm2(cast(int)x.length, x.ptr , incr);
}
/********************************************* Matrix Multiplication ******************************************/
/* 
  Matrix-Matrix multiplication - 
      works fine but we will use the implementation below this one
*/
Matrix!(T, layout) mult_old(T, CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA = CblasNoTrans, 
  CBLAS_TRANSPOSE transB = CblasNoTrans)
    (Matrix!(T, layout) a, Matrix!(T, layout) b)
if(isFloatingPoint!T)
{
  T alpha = 1;
  T beta = 0;
  int m = transA == CblasNoTrans ? cast(int)a.dim[0] : cast(int)a.dim[1];
  int n = transB == CblasNoTrans ? cast(int)b.dim[1] : cast(int)b.dim[0];
  T[] c; //set this length
  
  int k = transA == CblasNoTrans ? cast(int)a.dim[1] : cast(int)a.dim[0];
  
  int lda, ldb, ldc;
  if(transA == CblasNoTrans)
  	lda = layout == CblasRowMajor? k: m;
  else
  	lda = layout == CblasRowMajor? m: k;
  
  if(transB == CblasNoTrans)
  	ldb = layout == CblasRowMajor? n: k;
  else
  	ldb = layout == CblasRowMajor? k: n;
  
  ldc = layout == CblasRowMajor ? n : m;
  c.length = layout == CblasRowMajor ? ldc*m : ldc*n;
  
  dgemm(layout, transA, transB, m, n,
        k, alpha, a.getData.ptr, lda, b.getData.ptr, ldb,
        beta, c.ptr, ldc);
  return new Matrix!(T, layout)(c, [m, n]);
}

/* Matrix-Matrix multiplication */
Matrix!(T, layout) mult_(T, CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA = CblasNoTrans, 
  CBLAS_TRANSPOSE transB = CblasNoTrans)
    (Matrix!(T, layout) a, Matrix!(T, layout) b)
if(isFloatingPoint!T)
{
  T alpha = 1.;
  T beta = 0.;

  int m = transA == CblasNoTrans ? cast(int)a.nrow : cast(int)a.ncol;
  int k = transA == CblasNoTrans ? cast(int)a.ncol : cast(int)a.nrow;
  int n = transB == CblasNoTrans ? cast(int)b.ncol : cast(int)b.nrow;

  auto c = new T[m*n];

  int lda, ldb, ldc;
  if(transA == CblasNoTrans)
    lda = layout == CblasColMajor ? m : k;
  else
    lda = layout == CblasColMajor ? k : m;
  
  if(transB == CblasNoTrans)
    ldb = layout == CblasColMajor ? k : n;
  else
    ldb = layout == CblasColMajor ? n : k;
  
  ldc = layout == CblasColMajor ? m : n;

  dgemm(layout, transA, transB, m, n,
        k, alpha, a.getData.ptr, lda, b.getData.ptr, ldb,
        beta, c.ptr, ldc);
  
  return new Matrix!(T, layout)(c, [m, n]);
}

/* Matrix-Vector Multiplication */
ColumnVector!(T) mult_(T, CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA = CblasNoTrans)
    (Matrix!(T, layout) a, Vector!(T) x)
if(isFloatingPoint!T)
{
  T alpha = 1.0;
  T beta = 0.0;
  int m, n; int incx = 1; int incy = 1;
  m = cast(int)a.dim[0];
  n = cast(int)a.dim[1];
  T[] y;
  
  int lda;
  lda = layout == CblasRowMajor ? n : m;
  
  y.length = transA == CblasNoTrans ? m : n;
  
  dgemv(layout, transA, m, n, alpha, a.getData.ptr, lda, 
    x.getData.ptr, incx, beta, y.ptr, incy);

  return new ColumnVector!(T)(y);
}
/********************************************* Matrix Solver ******************************************/
/*
  Returns the solve of a matrix and a vector.

  TODO:
  1. Remember to implement the GLM solve function in such a way that 
  it tries the symmetrical method (where appropriate), then the general
  matrix, and then the pinv() function as a last resort.
*/
ColumnVector!(T) solve(CBLAS_SYMMETRY symmetry = CblasGeneral, T, CBLAS_LAYOUT layout)
(Matrix!(T, layout) mat, ColumnVector!(T) v){
  assert(mat.nrow == mat.ncol, "This solve function only works for square A matrices.");
  
	int p = cast(int)mat.nrow;
	int[] ipiv = new int[p];// ipiv.length = p;
  double[] b = v.getData.dup;
  T[] data = mat.data.dup;
  
  int info;
  static if(symmetry == CblasGeneral)
  {
    info = dgetrf(layout, p, p, data.ptr, p, ipiv.ptr);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function LAPACKE_dgetrf");
    info = dgetrs(layout, 'N', p, 1, data.ptr, p, ipiv.ptr, b.ptr, p);
    assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function LAPACKE_dgetrs");
  } else if(symmetry == CblasSymmetric){
    info = dpotrf(layout, 'U', p, data.ptr, p);
    assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function LAPACKE_dpotrf");
    info = dpotrs(layout, 'U', p, 1, data.ptr, p, b.ptr, p);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function LAPACKE_dpotrs");
  } else {
    assert(0, "Symmetry not recognised!");
  }
  return new ColumnVector!(T)(b);
}
/********************************************* Matrix Inverses ******************************************/
/* Returns the inverse of a matrix */
Matrix!(T, layout) inv(CBLAS_SYMMETRY symmetry = CblasGeneral, T, CBLAS_LAYOUT layout)(Matrix!(T, layout) mat){
	int p = cast(int)mat.nrow;
	int[] ipiv; ipiv.length = p;
  T[] data = mat.data.dup;
  
  int info;
  static if(symmetry == CblasGeneral)
  {
    info = dgetrf(layout, p, p, data.ptr, p, ipiv.ptr);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function LAPACKE_dgetrf");
    info = dgetri(layout, p, data.ptr, p, ipiv.ptr);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function LAPACKE_dgetri");
  } else if(symmetry == CblasSymmetric){
    info = dpotrf(layout, 'U', p, data.ptr, p);
    assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function LAPACKE_dpotrf");
    //writeln("dpotrf output:\n", data);
    info = dpotri(layout, 'U', p, data.ptr, p);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function LAPACKE_dpotri");
    //writeln("dpotri output:\n", data);
    /* Create regular square matrix from an upper triangular matrix */
    for(int j = 0; j < p; ++j)
    {
      for(int i = 0; i < j; ++i)
      {
        data[p*i + j] = data[p*j + i];
      }
    }
  } else {
    assert(0, "Symmetry not recognised!");
  }
  return new Matrix!(T, layout)(data, [p, p]);
}

/* Return the pseudo (generalized) inverse of a matrix */
Matrix!(T, layout) pinv(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) mat)
{
  assert(mat.nrow == mat.ncol, "Number of rows and columns of the matrix are not equal.");
	double[] a = mat.data.dup;
  int m = cast(int)mat.nrow;
  int info = 0; 
  auto s = new double[m];
  auto u = new double[m*m];
  auto vt = new double[m*m];
  auto superb = new double[m-1];
  int output = dgesvd(CblasColMajor, 'A', 'A', m, m, a.ptr, m, s.ptr, u.ptr, m, vt.ptr, m, superb.ptr );
  assert(info == 0, "LAPACKE_gesvd error: U" ~ info.stringof ~ 
        " is singular and its inverse could not be computed.");
  /* TODO: 
  ** Implement in the style of: 
  **   https://software.intel.com/en-us/articles/implement-pseudoinverse-of-a-matrix-by-intel-mkl
  */
  foreach(ref el; s)
  {
    if(el > 1E-9)
      el = 1/el;
  }
  auto V = new Matrix!(T, layout)(vt, [m, m]);
  return mult_!(T, layout, CblasTrans, CblasTrans)(
    sweep!((double x1, double x2) => x1 * x2)(V, s), 
    new Matrix!(T, layout)(u, [m, m]));
}
/********************************************* Accuracy ******************************************/
/* Epsilon for floats */
template eps(T)
if(isFloatingPoint!T)
{
  enum eps = T.epsilon;
}
template ceps(T)
if(isFloatingPoint!T)
{
  enum ceps = 1 - T.epsilon;
}
/******************************************* Link & Family Functions *********************************/
template initType(T)
{
  alias initType = Tuple!(T, T, T);
}
/* Probability Distributions */
abstract class AbstractDistribution(T)
{
  initType!(ColumnVector!T) init(Matrix!(T) _y, ColumnVector!T wts)
  {
    auto y = cast(ColumnVector!(T))_y;
    return tuple(y, y.dup, wts);
  }
  ColumnVector!T variance(ColumnVector!T mu);
  T variance(T mu);
  ColumnVector!T devianceResiduals(ColumnVector!T mu, ColumnVector!T y);
  T devianceResiduals(T mu, T y);
  T devianceResiduals(T mu, T y, T wts);
  ColumnVector!T devianceResiduals(ColumnVector!T mu, ColumnVector!T y, ColumnVector!T wts);
}
T y_log_y(T)(T y, T x)
{
  //pragma(inline, true);
  return y != 0 ? y * log(y/x) : 0;
}
class BinomialDistribution(T) : AbstractDistribution!(T)
{
  override initType!(ColumnVector!T) init(Matrix!T _y, ColumnVector!T wts)
  {
    ColumnVector!(T) y; ColumnVector!T mu;
    bool hasWeights = wts.len > 0;
    if(_y.ncol == 1)
    {
      y = cast(ColumnVector!(T))_y;
      if(wts.len == 0)
      {
        mu = map!( (T x) => (x + cast(T)0.5)/2 )(y);
      }else{
        mu = map!( (T x, T w) => (w * x + cast(T)0.5)/(w + 1) )(y, wts);
      }
    }else if(_y.ncol > 1)
    {
      y = new ColumnVector!(T)(_y.nrow);
      mu = new ColumnVector!(T)(_y.nrow);
      wts = new ColumnVector!(T)(_y.nrow);
      for(ulong i = 0; i < _y.nrow; ++i)
      {
        wts[i] = _y[i, 0] + _y[i, 1];
        y[i] = _y[i, 0]/wts[i];
        mu[i] = (wts[i] * y[i] + 0.5)/(wts[i] + 1);
      }
    }
    return tuple(y, mu, wts);
  }
  override T variance(T mu)
  {
    return mu * (1 - mu);
  }
  override ColumnVector!T variance(ColumnVector!T mu)
  {
    return map!( (T x) => variance(x) )(mu);
  }
  override T devianceResiduals(T mu, T y)
  {
    return 2*(y_log_y!(T)(y, mu) + y_log_y!(T)(1 - y, 1 - mu));
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!T mu, ColumnVector!T y)
  {
    return map!((T m, T x) => devianceResiduals(m, x))(mu, y);
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    return 2 * wts * (y_log_y!(T)(y, mu) + y_log_y!(T)(1 - y, 1 - mu));
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!T mu, ColumnVector!T y, ColumnVector!T wts)
  {
    return map!( (T m, T x, T w) => devianceResiduals(m, x, w) )(mu, y, wts);
  }
  override string toString()
  {
    return "BinomialDistribution";
  }
}
class PoissonDistribution(T) : AbstractDistribution!(T)
{
  override initType!(ColumnVector!T) init(Matrix!(T) _y, ColumnVector!T wts)
  {
    auto y = cast(ColumnVector!(T))_y;
    ColumnVector!(T) mu = map!( (T x) => x + 0.1 )(y);
    return tuple(y, mu, wts);
  }
  override T variance(T mu)
  {
    return mu;
  }
  override ColumnVector!(T) variance(ColumnVector!(T) mu)
  {
    return mu.dup;
  }
  override T devianceResiduals(T mu, T y)
  {
    T dev;
    if(y == 0)
      dev = 2 * mu;
    else if(y > 0)
      dev = 2 * (y * log(y/mu) - (y - mu));
    return dev;
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y)
  {
    return map!((T m, T x) => devianceResiduals(m, x))(mu, y);
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    T dev;
    if(y == 0)
      dev = 2 * wts * mu;
    else if(y > 0)
      dev = 2 * wts * (y * log(y/mu) - (y - mu));
    return dev;
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y, ColumnVector!(T) wts)
  {
    return map!((T m, T x, T w) => devianceResiduals(m, x, w))(mu, y, wts);
  }
  override string toString()
  {
    return "PoissonDistribution";
  }
}
class GaussianDistribution(T) : AbstractDistribution!(T)
{
  override T variance(T mu)
  {
    return cast(T)1;
  }
  override ColumnVector!(T) variance(ColumnVector!(T) mu)
  {
    return onesColumn!T(mu.len);
  }
  override T devianceResiduals(T mu, T y)
  {
    return (y - mu)^^2;
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    return wts * (y - mu)^^2;
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y)
  {
    return map!( (T m, T x) => devianceResiduals(m, x) )(mu, y);
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y, ColumnVector!(T) wts)
  {
    return map!( (T m, T x, T w) => devianceResiduals(m, x, w) )(mu, y, wts);
  }
  override string toString()
  {
    return "GaussianDistribution";
  }
}
class InverseGaussianDistribution(T) : AbstractDistribution!(T)
{
  override T variance(T mu)
  {
    return mu^^3;
  }
  override ColumnVector!(T) variance(ColumnVector!(T) mu)
  {
    return map!( (T m) => m^^3 )(mu);
  }
  override T devianceResiduals(T mu, T y)
  {
    return ((y - mu)^^2)/(y * (mu^^2));
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    return wts * ((y - mu)^^2)/(y * (mu^^2));
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y)
  {
    return map!( (T m, T x) => devianceResiduals(m, x) )(mu, y);
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y, ColumnVector!(T) wts)
  {
    return map!( (T m, T x, T w) => devianceResiduals(m, x, w) )(mu, y, wts);
  }
  override string toString()
  {
    return "InverseGaussianDistribution";
  }
}
class NegativeBinomialDistribution(T) : AbstractDistribution!(T)
{
  immutable(T) alpha; /* Parameter */
  override initType!(ColumnVector!(T)) init(Matrix!(T) _y, ColumnVector!(T) wts)
  {
    auto y = cast(ColumnVector!(T))_y;
    T tmp = 1/6;
    //auto mu = y.dup;
    //for(ulong i = 0; i < y.length; ++i)
    //{
    //  if(y[i] == 0)
    //    mu[i] += tmp;
    //}
    auto mu = map!( (T x) => x == 0 ? tmp : x)(y);
    return tuple(y, mu, wts);
  }
  override T variance(T mu)
  {
    return mu + alpha * (mu^^2);
  }
  override ColumnVector!(T) variance(ColumnVector!(T) mu)
  {
    return map!( (T m) => variance(m) )(mu);
  }
  override T devianceResiduals(T mu, T y)
  {
    T dev;
    T ialpha = alpha^^-1;
    if(y == 0)
      dev = 2 * ialpha * log(1/(1 + alpha*mu));
    else if(y > 0)
      dev = 2 * (y * log(y/mu) - (y + ialpha) * log((1 + alpha * y)/(1 + alpha * mu)));
    return dev;
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    T dev;
    T ialpha = alpha^^-1;
    if(y == 0)
      dev = 2 * wts * ialpha * log(1/(1 + alpha * mu));
    else if(y > 0)
      dev = 2 * wts * (y * log(y/mu) - (y + ialpha) * log((1 + alpha * y)/(1 + alpha * mu)));
    return dev;
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y)
  {
    return map!( (T m, T x) => devianceResiduals(m, x) )(mu, y);
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y, ColumnVector!(T) wts)
  {
    return map!( (T m, T x, T w) => devianceResiduals(m, x, w) )(mu, y, wts);
  }
  override string toString()
  {
    return "NegativeBinomialDistribution{alpha = " ~ to!string(alpha) ~ "}";
  }
  this(T _alpha)
  {
    alpha = _alpha;
  }
}
/*
  See p 149 & 433 Of Generalized Linear Models & Extensions, 
  by J. W. Hardin & J. M. Hilbe.
*/
class PowerDistribution(T) : AbstractDistribution!(T)
{
  T k;
  override T variance(T mu)
  {
    return mu^^k;
  }
  override ColumnVector!(T) variance(ColumnVector!(T) mu)
  {
    return map!( (T m) => variance(m) )(mu);
  }
  override T devianceResiduals(T mu, T y)
  {
    T ok = 1 - k;
    T tk = 2 - k;
    return ( (2 * y/( ok * ((y^^(ok)) - (mu^^ok)) )) - (2/( tk * ((y^^(tk)) - (mu^^tk)) )) );
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    T ok = 1 - k;
    T tk = 2 - k;
    return wts * ( (2 * y/( ok * ((y^^(ok)) - (mu^^ok)) )) - (2/( tk * ((y^^(tk)) - (mu^^tk)) )) );
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y)
  {
    return map!( (T m, T x) => devianceResiduals(m, x) )(mu, y);
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y, ColumnVector!(T) wts)
  {
    return map!( (T m, T x, T w) => devianceResiduals(m, x, w) )(mu, y, wts);
  }
  override string toString()
  {
    return "PowerDistribution{" ~ to!string(k) ~ "}";
  }
  this(T _k)
  {
    k = _k;
  }
}
class GammaDistribution(T) : AbstractDistribution!(T)
{
  override ColumnVector!T variance(ColumnVector!T mu)
  {
    return map!((T x) => x^^2)(mu);
  }
  override T variance(T mu)
  {
    return mu^^2;
  }
  override ColumnVector!T devianceResiduals(ColumnVector!T mu, ColumnVector!T y)
  {
    return map!((T m1, T y1) => 2*(((y1 - m1)/m1) - log(y1/m1)) )(mu, y);
  }
  override T devianceResiduals(T mu, T y)
  {
    return 2*( ((y - mu)/mu) - log(y/mu) );
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    return 2*wts*( ((y - mu)/mu) - log(y/mu) );
  }
  override ColumnVector!T devianceResiduals(ColumnVector!T mu, ColumnVector!T y, ColumnVector!T wts)
  {
    return map!((T m1, T y1, T wts1) => 2*wts1*(((y1 - m1)/m1) - log(y1/m1)) )(mu, y, wts);
  }
  override string toString()
  {
    return "GammaDistribution";
  }
}
/******************************************* Link Functions *********************************/
interface AbstractLink(T)
{
  ColumnVector!T linkfun(ColumnVector!T mu);
  T linkfun(T mu);
  ColumnVector!T deta_dmu(ColumnVector!T mu, ColumnVector!T eta);
  T deta_dmu(T mu, T eta);
  ColumnVector!T linkinv(ColumnVector!T eta);
  T linkinv(T eta);
  string toString();
}
class LogLink(T): AbstractLink!T
{
  T linkfun(T mu)
  {
    return log(mu);
  }
  ColumnVector!T linkfun(ColumnVector!T mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return mu^^-1;
  }
  ColumnVector!T deta_dmu(ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!((T m, T x) => deta_dmu(m, x))(mu, eta);
  }
  T linkinv(T eta)
  {
    return exp(eta);
  }
  ColumnVector!T linkinv(ColumnVector!T eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "LogLink";
  }
}
class IdentityLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return mu;
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return mu.dup;
  }
  T deta_dmu(T mu, T eta)
  {
    return 1;
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return fillColumn!(T)(1, eta.len);
  }
  T linkinv(T eta)
  {
    return eta;
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return eta.dup;
  }
  override string toString()
  {
    return "IdentityLink";
  }
}
class InverseLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return mu^^-1;
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => x^^-1 )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return -(mu^^-2);
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T t) => deta_dmu(m, t) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return eta^^-1;
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T t) => t^^-1 )(eta);
  }
  override string toString()
  {
    return "InverseLink";
  }
}
class LogitLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return log(mu/(1 - mu));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T m) => linkfun(m) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return (mu * (1 - mu))^^-1;
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    T x = exp(eta);
    return x/(1 + x);
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "LogitLink";
  }
}
class CauchitLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return tan(pi * (mu - 0.5));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T m) => linkfun(m) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return pi * (cos(pi * (mu - 0.5))^^(-2));
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T t) => deta_dmu(m, t) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return (atan(eta)/pi) + 0.5;
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "CauchitLink";
  }
}
class ProbitLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return qnorm(mu);
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!(qnorm)(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return dnorm(eta)^^-1;
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!(dnorm)(eta);
  }
  T linkinv(T eta)
  {
    return pnorm(eta);
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "ProbitLink";
  }
}
class PowerLink(T) : AbstractLink!(T)
{
  immutable(T) alpha;
  LogLink!T logl;
  T linkfun(T mu)
  {
    return alpha == 0 ? logl.linkfun(mu) : mu^^alpha;
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return alpha == 0 ? logl.deta_dmu(mu, eta) : alpha * (mu^^(alpha - 1));
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return alpha == 0 ? logl.linkinv(eta) : eta^^(alpha^^-1);
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "PowerLink{" ~ to!string(alpha) ~ "}";
  }
  this(T _alpha)
  {
    alpha = _alpha;
    logl = new LogLink!T();
  }
}
class OddsPowerLink(T) : AbstractLink!(T)
{
  immutable(T) alpha;
  LogitLink!(T) logit;
  T linkfun(T mu)
  {
    return alpha == 0 ? logit.linkfun(mu) : ((mu/(1 - mu))^^alpha - 1)/alpha;
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return alpha == 0 ? logit.deta_dmu(mu, eta) : (mu^^(alpha - 1))/((1 - mu)^^(alpha + 1));
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    T ret;
    if(alpha == 0)
    {
      return logit.linkinv(eta);
    }else{
      T tmp = ((eta * alpha + 1)^^(1/alpha));
      ret = min(max(tmp/(1 + tmp), eps!(T)), ceps!(T));
    }
    return ret;
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "OddsPowerLink";
  }
  this(T _alpha)
  {
    alpha = _alpha;
    logit = new LogitLink!(T)();
  }
}
class LogComplementLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return log(max(1 - mu, eps!(T)));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return -(max(1 - mu, eps!(T)))^^(-1);
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return min(max(-expm1(eta), eps!(T)), ceps!(T));
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "LogComplementLink";
  }
}
class LogLogLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return -log(-log(mu));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return -(mu * log(mu))^^-1;
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return exp(-exp(-eta));
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "LogLogLink";
  }
}
class ComplementaryLogLogLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return log(-log(1 - mu));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return ((mu - 1) * log(1 - mu))^^-1;
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return 1 - exp(-exp(eta));
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "LogLogLink";
  }
}
class NegativeBinomialLink(T) : AbstractLink!(T)
{
  immutable(T) alpha;
  T linkfun(T mu)
  {
    T ialpha = alpha^^-1;
    return log(mu/(mu + ialpha));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return 1/(mu + alpha * mu^^2);
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    T tmp = exp(eta);
    return tmp/(alpha * (1 - tmp));
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "NegativeBinomialLink{"~ to!string(alpha) ~ "}";
  }
  this(T _alpha)
  {
    alpha = _alpha;
  }
}
/******************************************* Weight & Systematic Component *********************************/
auto Z(T)(AbstractLink!T link, T y, T mu, T eta)
{
  return link.deta_dmu(mu, eta) * (y - mu) + eta;
}
auto Z(T)(AbstractLink!T link, ColumnVector!T y, 
          ColumnVector!T mu, ColumnVector!T eta)
{
  return map!( (T x, T m, T t) => link.Z(x, m, t) )(y, mu, eta);
}

/* Weights Vector */
auto W(T)(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
{
  return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^-1;
}
auto W(T)(AbstractDistribution!T distrib, AbstractLink!T link, 
          ColumnVector!T mu, ColumnVector!T eta)
{
  return map!( (T m, T x) => W!(T)(distrib, link, m, x) )(mu, eta);
}
/* Square Root of Weights Vector */
auto WS(T)(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
{
  return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^-0.5;
}
auto WS(T)(AbstractDistribution!T distrib, AbstractLink!T link,
          ColumnVector!T mu, ColumnVector!T eta)
{
  return map!( (T m, T x) => WS!(T)(distrib, link, m, x) )(mu, eta);
}
/******************************************* QR Decomposition Functions *********************************/
/*
  Convert outcome from QR algorithm to R Matrix. See page 723 from the 
  MKL library. Table "Computational Routines for Orthogonal Factorization".
*/
auto qrToR(T, CBLAS_LAYOUT layout = CblasColMajor)(Matrix!(T, layout) qr)
{
  ulong n = qr.ncol * qr.ncol;
  auto R = matrix(new T[n], qr.ncol);
  for(ulong i = 0; i < qr.ncol; ++i)
  {
    for(ulong j = 0; j < qr.ncol; ++j)
    {
      T tmp = i <= j ? qr[i, j] : 0;
      R[i, j] = tmp;
      //writeln("t(i, j)", "(", i, ", ", j, "): ", tmp);
    }
  }
  return R;
}
/* Least Squares QR Decomposition */
auto qrls(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) X, ColumnVector!(T) y)
{
  int m = cast(int)X.nrow;
  int n = cast(int)X.ncol;
  assert(m > n, "Number of rows is less than the number of columns.");
  auto a = X.getData.dup;
  T[] tau = new T[n];
  int lda = layout == CblasColMajor ? m : n;
  int info = dgeqrf(layout, m, n, a.ptr, lda, tau.ptr);
  
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function LAPACKE_dgeqrf");
  //writeln("tau:", tau);
  auto Q = matrix(a, [m, n]);
  auto R = qrToR(Q);
  info = dorgqr(layout, m, n, n, a.ptr, lda, tau.ptr);
  
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function LAPACKE_dorgqr");
  //writeln("Q Matrix:\n", Q);
  //writeln("R Matrix:\n", R);

  ColumnVector!(T) z = mult_!(T, layout, CblasTrans)(Q, y);
  //writeln("z: \n", z);
  ColumnVector!(T) coef = solve(R, z);
  //auto ret = tuple!("coef", "R")(coef, R);
  //writeln("Coefficient & R:\n", ret);
  return tuple!("coef", "R")(coef, R);
}
/******************************************* Internal Solver Fucntions *********************************/
/*
  Conventional Solver:
  coef = (X^TWX)^(-1) (X^T W y)
*/
void _conventional_solver(T, CBLAS_LAYOUT layout = CblasColMajor)
        (ref Matrix!(T, layout) xwx, ref Matrix!(T, layout) x, ref ColumnVector!(T) z, 
        ref ColumnVector!(T) w, ref ColumnVector!(T) coef)
{
  //writeln("Conventional Solver");
  auto xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
  xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(xw, x);
  auto xwz = mult_!(T, layout, CblasTrans)(xw, z);
  coef = solve(xwx, xwz);
  return;
}
/*
  QR Solver:
  coef = (R)^(-1) (Q^T y)
*/
void _qr_solver(T, CBLAS_LAYOUT layout = CblasColMajor)
        (ref Matrix!(T, layout) R, ref Matrix!(T, layout) x, 
        ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
        ref ColumnVector!(T) coef)
{
  //writeln("QR Solver");
  auto xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
  auto zw = map!( (x1, x2) => x1 * x2 )(z, w);
  auto coefR = qrls!(T, layout)(xw, zw);
  coef = coefR.coef;
  R = coefR.R;
}
/******************************************* Error Functions *********************************/
alias fabs abs;

T absoluteError(T)(T x, T y)
if(isFloatingPoint!T)
{
  return abs(x - y);
}
T absoluteError(T)(Vector!T x, Vector!T y)
if(isFloatingPoint!T)
{
  assert(x.len == y.len, "vectors are not equal.");
  T ret = 0;
  for(ulong i = 0; i < x.len; ++i)
    ret += (x[i] - y[i])^^2;
  return (ret^^0.5);
}
T relativeError(T)(T x, T y)
if(isFloatingPoint!T)
{
  return abs(x - y)/(0.1 + abs(x));
}
T relativeError(T)(Vector!T x, Vector!T y)
if(isFloatingPoint!T)
{
  assert(x.len == y.len, "vectors are not equal.");
  T x1 = 0; T x2 = 0;
  for(ulong i = 0; i < x.len; ++i)
  {
    x1 += (x[i] - y[i])^^2;
    x2 += x[i]^^2;
  }
  return (x1^^0.5)/(1E-5 + (x2^^0.5));
}
/****************************************** Control Class *****************************************/
class Control(T)
if(isFloatingPoint!T)
{
  immutable(T) epsilon;
  immutable(int) maxit;
  immutable(bool) printError;
  immutable(bool) printCoef;
  immutable(T) minstep;
  this(T _epsilon = 1E-7, int _maxit = 25, 
      bool _printError = false, bool _printCoef = false, 
      T _minstep = 1E-5)
  {
    epsilon = _epsilon; maxit = _maxit;
    printError = _printError; printCoef = _printCoef;
    minstep = _minstep;
  }
  override string toString() const
  {
    string repr = "Control(T = " ~ T.stringof ~ ")\n{\n";
    repr ~= "  Epsilon = " ~ to!string(epsilon) ~ ";\n";
    repr ~= "  Maxit = " ~ to!string(maxit) ~ ";\n";
    repr ~= "  Print Error = " ~ to!string(printError) ~ ";\n";
    repr ~= "  Minstep = " ~ to!string(printCoef) ~ ";\n";
    repr ~= "}\n";
    return repr;
  }
}
/**************************************** GLM Result Class ***************************************/
class GLM(T, CBLAS_LAYOUT L)
if(isFloatingPoint!T)
{
  ulong niter;
  bool converged;
  AbstractDistribution!T distrib;
  AbstractLink!T link;
  T[] coefficients;
  T[] standardError;
  Matrix!(T, L) cov;
  T deviance;
  T absoluteError;
  T relativeError;
  this(T, CBLAS_LAYOUT L)(ulong _niter, bool _converged, AbstractDistribution!T _distrib, AbstractLink!T _link,
      ColumnVector!T coeff, Matrix!(T, L) _cov, T _deviance, T absErr, T relErr)
  {
    niter = _niter;
    converged = _converged;
    distrib = _distrib;
    link = _link;
    coefficients = coeff.getData;
    standardError = new T[_cov.nrow];
    for(ulong i = 0; i < _cov.nrow; ++i)
      standardError[i] = _cov[i, i]^^0.5;
    cov = _cov;
    deviance = _deviance;
    absoluteError = absErr;
    relativeError = relErr;
  }
  override string toString()
  {
    string rep = "GLM(" ~ link.toString() ~ ", " ~ distrib.toString() ~ ")\n";
    rep ~= "Info(Converged = " ~ to!string(converged) ~ ", Iterations = " ~ to!string(niter) ~ ")\n";
    rep ~= "Error(Absolute Error = " ~ to!string(absoluteError) ~
            ", Relative Error = " ~ to!string(relativeError) ~ 
            ", Deviance = " ~ to!string(deviance) ~ ")\n";
    rep ~= "Coefficients:\n" ~ to!string(coefficients) ~ "\n";
    rep ~= "StandardError:\n" ~ to!string(standardError) ~ "\n";
    return rep;
  }
}
/**************************************** GLM Function ***************************************/
auto glm(T, CBLAS_LAYOUT layout = CblasColMajor)(Matrix!(T, layout) x, 
        Matrix!(T, layout) _y, AbstractDistribution!T distrib, AbstractLink!T link,
        ColumnVector!T offset = zerosColumn!T(0), ColumnVector!T weights = zerosColumn!T(0), 
        Control!T control = new Control!T(), bool qrSolver = false)
if(isFloatingPoint!T)
{
  auto init = distrib.init(_y, weights);
  auto y = init[0]; auto mu = init[1]; weights = init[2];
  auto eta = link.linkfun(mu);

  auto coef = zerosColumn!T(x.ncol);
  auto coefold = zerosColumn!T(x.ncol);

  auto absErr = T.infinity;
  auto relErr = T.infinity;
  auto residuals = zerosColumn!T(y.len);
  auto dev = T.infinity;
  auto devold = T.infinity;

  ulong iter = 1;
  auto n = x.nrow; auto p = x.ncol;
  bool converged, badBreak, doOffset, doWeights;

  if(offset.len != 0)
    doOffset = true;
  if(weights.len != 0)
    doWeights = true;

  Matrix!(T, layout) cov, xwx, R;
  ColumnVector!(T) w, wy;
  while(relErr > control.epsilon) /* (absErr > control.epsilon) | (relErr > control.epsilon) */
  {
    if(control.printError)
      writefln("Iteration: %d", iter);
    auto z = link.Z(y, mu, eta);
    if(doOffset)
      z = map!( (x1, x2) => x1 - x2 )(z, offset);
    
    /* Weights calculation standard vs sqrt */
    if(qrSolver)
    {
      w = WS(distrib, link, mu, eta);
    }else{
      w = W(distrib, link, mu, eta);
    }
    
    if(doWeights)
      w = map!( (x1, x2) => x1*x2 )(w, weights);
    
    /* 
      Make sure that you are doing this calculation somewhere else 
    */
    //auto xw = x.dup;
    //sweepRef!( (x1, x2) => x1 * x2 )(xw, w);

    //xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(xw, x);
    //auto xwz = mult_!(T, layout, CblasTrans)(xw, z);
    //coef = solve(xwx, xwz);

    //writeln("Debug point 1");

    if(qrSolver)
    {
      _qr_solver!(T, layout)(R, x, z, w, coef);
    }else{
      _conventional_solver!(T, layout)(xwx, x, z, w, coef);
    }
    
    //writeln("Debug point 2");
    
    if(control.printCoef)
      writeln(coef);
    
    eta = mult_(x, coef);
    if(doOffset)
      eta += offset;
    mu = link.linkinv(eta);

    if(weights.len == 0)
      residuals = distrib.devianceResiduals(mu, y);
    else
      residuals = distrib.devianceResiduals(mu, y, weights);
    
    dev = sum!T(residuals);

    absErr = absoluteError(dev, devold);
    relErr = relativeError(dev, devold);

    T frac = 1;
    auto coefdiff = map!( (x1, x2) => x1 - x2 )(coef, coefold);

    //Step Control
    while(dev > (devold + control.epsilon*dev))
    {
      if(control.printError)
      {
        writeln("\tStep control");
        writeln("\tFraction: ", frac);
        writeln("\tDeviance: ", dev);
        writeln("\tAbsolute Error: ", absErr);
        writeln("\tRelative Error: ", relErr);
      }
      frac *= 0.5;
      coef = map!( (T x1, T x2) => x1 + (x2 * frac) )(coefold, coefdiff);

      if(control.printCoef)
        writeln(coef);
      
      eta = mult_(x, coef);
      if(doOffset)
        eta += offset;
      mu = link.linkinv(eta);

      if(weights.len == 0)
        residuals = distrib.devianceResiduals(mu, y);
      else
        residuals = distrib.devianceResiduals(mu, y, weights);
      
      dev = sum!T(residuals);

      absErr = absoluteError(dev, devold);
      relErr = relativeError(dev, devold);

      if(frac < control.minstep)
        assert(0, "Step control exceeded.");
    }
    devold = dev;
    coefold = coef.dup;

    if(control.printError)
    {
      writeln("\tDeviance: ", dev);
      writeln("\tAbsolute Error: ", absErr);
      writeln("\tRelative Error: ", relErr);
    }
    if(iter >= control.maxit)
    {
      writeln("Maximum number of iterations " ~ to!string(control.maxit) ~ " has been reached.");
      badBreak = true;
      break;
    }
    iter += 1;
    //writeln("Absolute Error: ", absErr);
    //writeln("Relative Error: ", relErr);
  }
  if(badBreak)
    converged = false;
  else
    converged = true;
  
  //writeln("Debug point 3");
  if(qrSolver)
  {
    //writeln("Coefficient:\n", coef.getData);
    //writeln("Dimension of xwx: (", R.nrow, ", ", R.ncol, ")");
    xwx = mult_!(T, layout, CblasTrans)(R, R.dup);
    cov = inv(xwx);
  }else{
    cov = inv(xwx);
  }
  auto obj = new GLM!(T, layout)(iter, converged, distrib, link, coef, cov, dev, absErr, relErr);
  return obj;
}
/**************************************** BINARY IO ***************************************/
void writeColumnVector(T)(string fileName, ColumnVector!T v)
{
  auto file = File(fileName, "wb");
  size_t[1] n = [v.len];
  file.rawWrite(n);
  file.rawWrite(v.getData);
  return;
}
void writeRowVector(T)(string fileName, RowVector!T v)
{
  auto file = File(fileName, "wb");
  size_t[1] n = [v.len];
  file.rawWrite(n);
  file.rawWrite(v.getData);
  return;
}
ColumnVector!T readColumnVector(T)(string fileName)
{
  auto file = File(fileName, "rb");
  size_t[1] n;
  file.rawRead(n);
  auto vec = new T[n[0]];
  file.rawRead(vec);
  return new ColumnVector!T(vec);
}
RowVector!T readRowVector(T)(string fileName)
{
  auto file = File(fileName, "rb");
  size_t[1] n;
  file.rawRead(n);
  auto vec = new T[n[0]];
  file.rawRead(vec);
  return new RowVector!T(vec);
}

/* Function to read 2D Array */
void writeMatrix(T, CBLAS_LAYOUT layout = CblasColMajor)(string fileName, Matrix!(T, layout) mat)
{
  auto file = File(fileName, "wb");
  size_t[2] dim = [mat.nrow, mat.ncol];
  file.rawWrite(dim);
  file.rawWrite(mat.getData);
  return;
}
Matrix!(T, layout) readMatrix(T, CBLAS_LAYOUT layout = CblasColMajor)(string fileName)
{
  auto file = File(fileName, "rb");
  size_t[2] dim;
  file.rawRead(dim);
  auto mat = new T[dim[0]*dim[1]];
  file.rawRead(mat);
  return new Matrix!(T, layout)(mat, [cast(ulong)dim[0], cast(ulong)dim[1]]);
}
/*************************************************************************************/
void demo1()
{
  double[] dat1 = [0.5258874319129798,    0.1748310792322596, 0.32741218855864074, 
                   0.27457761265628555,   0.5884570435236942, 0.24725859282363394, 
                   0.0026890474662464303, 0.9497754886400656, 0.02207037565505421, 
                   0.6907347285327676,    0.9592865249385867, 0.0037546990281474013, 
                   0.5889903715624345,    0.9394951355167158, 0.4691847847916524, 
                   0.6715916314231278,    0.7554381258134812, 0.9471071671056135, 
                   0.5866722794791475,    0.8811154762774951];
  auto m1 = new Matrix!double(dat1, 5, 4);
  m1.writeln;
  int[] dat2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
  auto m2 = new Matrix!int(dat2, 5, 4);
  m2.writeln;

  double[] dat3 = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
  auto m3 = new Matrix!double(dat3, 5, 4);
  m3.writeln;
  m3[2, 2] = 3.142;
  writeln("Change matrix index at m[2, 2]:\n", m3);
  //writeln("data:\n", m3.data);

  double[] dat4 = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9];
  auto v1 = new ColumnVector!double(dat4);
  v1.writeln;
  v1[3] = 3.142;
  writeln("Change index in the matrix:\n", v1);

  double[] dat5 = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9];
  auto v2 = new RowVector!double(dat5);
  v2.writeln;
  writeln("v2[3]: ", v2[3]);
  v2[3] = 3.142;
  writeln("Change index in the matrix:\n", v2);

  auto m6 = new Matrix!double([1.0, 2.0, 3.0, 4.0, 5.0, 
                     6.0, 7.0, 8.0, 9.0, 10.0,
                     11.0, 12.0, 13.0, 14.0, 15.0], 5, 3);
  auto v4 = new ColumnVector!double([1.0, 2.0, 3.0, 4.0, 5.0]);
  writeln("Sweep multiplication of \n", m6, "and\n", v4);
  writeln("Sweep multiplication of matrix and array \n", sweep!((double x1, double x2) => x1*x2, Column)(m6, v4.getData));

  writeln("Outcome of Column-wise matrix-vector multiplication sweep function\n", 
    sweep!((double x1, double x2) => x1*x2, Column)(m6, v4));
  auto v5 = new RowVector!double([1.0, 2.0, 3.0]);
  writeln("Outcome of Row-wise matrix-vector multiplcation sweep function\n", 
    sweep!((double x1, double x2) => x1*x2, Row)(m6, v5));
  
  auto m7 = new Matrix!double([16.0, 17.0, 18.0, 19.0, 20.0,
                               21.0, 22.0, 23.0, 24.0, 25.0,
                               26.0, 27.0, 28.0, 29.0, 30.0], 5, 3);
  writeln("Sweep function for two matrices\n", sweep!((double x1, double x2) => x1 * x2)(m6, m7));
  // double[5] arr1 = 1.0; //initialization example
  auto m8 = new Matrix!double([16.0, 17.0, 18.0, 19.0, 20.0,
                               21.0, 22.0, 23.0, 24.0, 25.0,
                               26.0, 27.0], 4, 3);
  
  /* This results in an error because the matrices have different dimensions */
  //writeln("This should be an error: ", sweep!((double x1, double x2) => x1 + x2)(m7, m8));

  dispatchTemplate(m8);

  /* Create a matrix using the array to mass on the matrix type */
  double[] arr = [1.0, 2.0, 3.0, 4.0];
  /* Testing the type inference for matrix constructor */
  auto m9 = matrix(arr, 2, 2);
  writeln("Type inferred constructed matrix: \n", m9);

  auto m10 = matrix(m9);

  writeln("Matrix multiplication: \n", mult_(m9, m10));
  writeln("Matrix multiplication: \n", mult_!(double, CblasColMajor, CblasNoTrans, CblasTrans)(m9, m10));

  auto m11 = matrix(m7);

  writeln("Original Matrix: ", m7);
  writeln("Transpose: ", m7.t());

  auto m13 = createRandomMatrix(5);
  writeln("Original Matrix: \n", m13);
  writeln("Transpose of square matrix: ", m13.t());

  auto v6 = columnVector([1.0, 2.0, 3.0]);

  auto v7 = mult_(m7, v6);
  writeln("Output of Matrix-Vector multiplication:\n", v7);

  auto v8 = columnVector([6.0, 7.0, 8.0, 9.0, 10.0]);
  writeln("Map function for column vector: \n", map!((double x1, double x2) => x1*x2)(v4, v8));

  auto v9 = rowVector([1.0, 2.0, 3.0, 4.0, 5.0]);
  auto v10 = rowVector([6.0, 7.0, 8.0, 9.0, 10.0]);
  writeln("Map function for row vector:\n", map!((double x1, double x2) => x1*x2)(v9, v10));

  writeln("Map function for column vector:\n", map!((double v) => v^^2)(v8));
  writeln("Map function for row vector:\n", map!((double v) => v^^2)(v9));

  auto m12 = createRandomMatrix(5);
  writeln("Create random square matrix:\n", m12);
  writeln("Inverse of a square matrix:\n", inv(m12));
  writeln("Pseudo-inverse of a square matrix:\n", pinv(m12));
  writeln("Create random rectangular matrix:\n", createRandomMatrix(7, 3));

  writeln("Create random column vector:\n", createRandomColumnVector(5));
  writeln("Create random row vector:\n", createRandomRowVector(5));

  //auto sm1 = createSymmetricMatrix!double(9);
  double[] arr2 = [30, 1998, 1594, 1691, 1939, 2243, 1288, 1998, 138208, 108798, 
                   115325, 131824, 150101,  86673, 1594, 108798,  89036,  91903, 
                   104669, 119695,  69689, 1691, 115325,  91903,  99311, 111561, 
                   126821,  74462,   1939, 131824, 104669, 111561, 128459, 146097,  
                   85029, 2243, 150101, 119695, 126821, 146097, 170541,  97136, 
                   1288, 86673, 69689, 74462, 85029, 97136, 58368];
  auto sm1 = matrix(arr2, [7, 7]);
  writeln("Create random symmetric matrix:\n", sm1);
  writeln("General inverse of symmetric matrix:\n", inv(sm1));
  writeln("Symmetric inverse of symmetric matrix:\n", inv!(CblasSymmetric)(sm1));

  double[] arr3 = [477410, 32325450, 25832480, 27452590, 31399180, 36024970, 20980860];
  auto cv1 = columnVector(arr3);
  writeln("Matrix solve for general matrices: \n", solve(sm1, cv1));
  writeln("Matrix solve for symmetric matrices:\n", solve!(CblasSymmetric)(sm1, cv1));

  writeln("Epsilon: ", to!string(eps!(double)), ", Compliment Epsilon: ", to!string(ceps!(double)), "\n");
  writeln(new Control!(double)());

  writeln("Norm [1:6]:\n", norm([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
  
  auto v11 = columnVector(createRandomArray(5));
  auto v12 = columnVector(createRandomArray(5));

  writeln("Array 1: ", v11);
  writeln("Array 2: ", v12);
  writeln("absoluteError(ColumnVector1, ColumnVector2): ", absoluteError(v11, v12));
  writeln("relativeError(ColumnVector1, ColumnVector2): ", relativeError(v11, v12));

  writeln("Write this column vector to file:\n", v11);
  writeColumnVector("ColumnVector.bin", v11);
  auto v13 = readColumnVector!double("ColumnVector.bin");
  writeln("Read Column Vector from file:\n", v13);
  "ColumnVector.bin".remove();

  writeln("Write this row vector to file:\n", v9);
  writeRowVector("RowVector.bin", v9);
  auto v14 = readRowVector!double("RowVector.bin");
  writeln("Read Row Vector from file:\n", v14);
  "RowVector.bin".remove();

  auto m14 = createRandomMatrix(7, 3);
  writeln("Matrix to be written to file:\n", m14);
  writeMatrix("Matrix.bin", m14);
  string xFile = "Matrix.bin";
  auto m15 = readMatrix!double(xFile);
  writeln("Matrix read from file:\n", m15);
  xFile.remove();

  return;
}

void qr_test()
{
  auto X = matrix!double([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1], [20, 2]);
  auto y = columnVector!double([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 
    5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 
    4.89, 4.32, 4.69]);
  auto qrOutput = qrls(X, y);
  writeln("QR decomposition Coefficient: ", qrOutput.coef);
  writeln("QR decomposition R: ", qrOutput.R);
}

void qr_vs_conventional()
{
  /* GLM Demo */

  /* Data Load */
  string path = "/home/chib/code/GLMPrototype/";
  auto energyX = readMatrix!double(path ~ "data/energyX.bin");
  auto energyY = readMatrix!double(path ~ "data/energyY.bin");

  /* Gamma Distribution With Log Link */
  import std.datetime.stopwatch : AutoStart, StopWatch;
  auto sw = StopWatch(AutoStart.no);
  sw.start();
  auto gamma_distrib_log_link = glm(energyX, energyY, 
      new GammaDistribution!double(), new LogLink!double());
  sw.stop();
  writeln(gamma_distrib_log_link);
  writeln("Time taken: ", sw.peek.total!"msecs");

  return;
}


void testMatrixVectorConversions()
{
  auto mat1 = createRandomMatrix(10, 1); // Column Matrix
  writeln("Column matrix: \n", mat1);

  auto vec = cast(ColumnVector!double)mat1;
  writeln("Converted to column vector: \n", vec);

  vec[0] = 99;
  writeln("Changed first item in the vector to 99, original: \n", mat1);

  /* Now cast to column vector */
  auto mat2 = cast(Matrix!(double))vec;
  writeln("Cast back to matrix from vector: ", mat2);

  /* Convert matrix to row vector */
  auto vec2 = cast(RowVector!(double))mat2;
  writeln("Cast matrix to row vector: ", vec2);
}


void main()
{
  /* GLM Demo */

  /* Data Load */
  string path = "/home/chib/code/GLMPrototype/";
  auto energyX = readMatrix!double(path ~ "data/energyX.bin");
  auto energyY = readMatrix!double(path ~ "data/energyY.bin");

  /* Insurance data */
  auto insuranceX = readMatrix!double(path ~ "data/insuranceX.bin");
  auto insuranceY = readMatrix!double(path ~ "data/insuranceY.bin");
  
  /* Credit Card Fraud */
  auto creditX = readMatrix!double(path ~ "data/creditX.bin");
  auto creditY = readMatrix!double(path ~ "data/creditY.bin");
  
  /* GPA Data */
  auto gpaX = readMatrix!double(path ~ "data/gpaX.bin");
  auto gpaY = readMatrix!double(path ~ "data/gpaY.bin");
  
  /* Cars Data */
  auto carsX = readMatrix!double(path ~ "data/carsX.bin");
  auto carsY = readMatrix!double(path ~ "data/carsY.bin");
  
  /* Quine Data for negative Binomial Distribution */
  auto quineX = readMatrix!double(path ~ "data/quineX.bin");
  auto quineY = readMatrix!double(path ~ "data/quineY.bin");

  /* Education Data for negative Binomial Distribution */
  auto educationX = readMatrix!double(path ~ "data/educationX.bin");
  auto educationY = readMatrix!double(path ~ "data/educationY.bin");

  if(false)
  {
  /* Gamma Distribution With Log Link */
  auto gamma_distrib_log_link = glm(energyX, energyY, new GammaDistribution!double(), new LogLink!double());
  writeln(gamma_distrib_log_link);
  
  /* Gamma Distribution With Inverse Link */
  auto gamma_distrib_inv_link = glm(energyX, energyY, new GammaDistribution!double(), new InverseLink!double());
  writeln(gamma_distrib_inv_link);
  
  /* Gamma Distribution With Identity Link */
  auto gamma_distrib_identity_link = glm(energyX, energyY, new GammaDistribution!double(), new IdentityLink!double());
  writeln(gamma_distrib_identity_link);
  
  /* Gamma Distribution With Power Link */
  auto gamma_distrib_power_link = glm(energyX, energyY, new GammaDistribution!double(), new PowerLink!double(0));
  writeln(gamma_distrib_power_link);
  auto gamma_distrib_power_link_2 = glm(carsX, carsY, new GammaDistribution!double(), new PowerLink!double(1/3));
  writeln(gamma_distrib_power_link_2);

  /* Gamma Distribution With Negative Binomial Link */
  auto gamma_distrib_negative_binomial_link_1 = glm(carsX, carsY, new GammaDistribution!double(), new NegativeBinomialLink!double(1.0));
  writeln(gamma_distrib_negative_binomial_link_1);
  auto gamma_distrib_negative_binomial_link_2 = glm(energyX, energyY, new GammaDistribution!double(), new NegativeBinomialLink!double(2.0));
  writeln(gamma_distrib_negative_binomial_link_2);
  /* Binomial Distribution With Logit Link Function */
  auto binomial_logit_link = glm(creditX, creditY, 
      new BinomialDistribution!double(), new LogitLink!double());
  writeln(binomial_logit_link);
  openblas_set_num_threads(1); /* Set the number of BLAS threads */
  /* Binomial Distribution With Probit Link Function */
  auto binomial_probit_link = glm(gpaX, gpaY, 
      new BinomialDistribution!double(), new ProbitLink!double());
  writeln(binomial_probit_link);
  /* Binomial Distribution With CauchitLink Function */
  auto binomial_cauchit_link = glm(gpaX, gpaY, 
      new BinomialDistribution!double(), new CauchitLink!double());
  writeln(binomial_cauchit_link);
  /* Binomial Distribution With OddsPowerLink Function */
  auto binomial_oddspower_link = glm(creditX, creditY, 
      new BinomialDistribution!double(), new OddsPowerLink!double(1));
  writeln(binomial_oddspower_link);
  /* Binomial Distribution With LogComplementLink Function */
  auto binomial_logcomplement_link = glm(gpaX, gpaY, 
      new BinomialDistribution!double(), new LogComplementLink!double());
  writeln(binomial_logcomplement_link);
  /* Binomial Distribution With LogLogLink Function */
  auto binomial_loglog_link = glm(gpaX, gpaY, 
      new BinomialDistribution!double(), new LogLogLink!double());
  writeln(binomial_loglog_link);
  /* Binomial Distribution With ComplementaryLogLogLink Function */
  auto binomial_complementaryloglog_link = glm(gpaX, gpaY, 
      new BinomialDistribution!double(), new ComplementaryLogLogLink!double());
  writeln(binomial_complementaryloglog_link);

  /* Now Test Different Distributions With Specific Link Functions */
  /* LogLink With Gaussian Distribution */
  auto log_link_gaussian_distrib = glm(energyX, energyY, 
      new GaussianDistribution!double(), new LogLink!double());
  writeln(log_link_gaussian_distrib);
  /* LogLink With Gamma Distribution */
  auto log_link_gamma_distrib = glm(energyX, energyY, 
      new GammaDistribution!double(), new LogLink!double());
  writeln(log_link_gamma_distrib);
  /* LogLink With InverseGaussian Distribution */
  auto log_link_inversegaussian_distrib = glm(energyX, energyY, 
      new InverseGaussianDistribution!double(), new LogLink!double());
  writeln(log_link_inversegaussian_distrib);
  /* LogLink With Poisson Distribution */
  auto log_link_poisson_distrib = glm(energyX, energyY, 
      new PoissonDistribution!double(), new LogLink!double());
  writeln(log_link_poisson_distrib);

  /* LogitLink With Binomial Distribution */
  auto logit_link_binomial_distrib = glm(creditX, creditY, 
      new BinomialDistribution!double(), new LogitLink!double());
  writeln(logit_link_binomial_distrib);
  /* LogitLink With Negative Binomial Distribution */
  auto logit_link_negative_binomial_distrib = glm(energyX, energyY, 
      new NegativeBinomialDistribution!double(0.5), new LogLink!double());
  writeln(logit_link_negative_binomial_distrib);
  /* LogLink With Power Distribution */
  auto log_link_power_distrib = glm(carsX, carsY, 
      new PowerDistribution!double(0.5), new PowerLink!double(0.5));
  writeln(log_link_power_distrib);
  /* Logit Link With Binomial Distribution - Works fine */
  auto logit_link_binomial_distrib_two_col = glm(educationX, educationY, 
      new BinomialDistribution!double(), new LogitLink!double());
  writeln(logit_link_binomial_distrib_two_col);
  }
  /* Cauchit Link With Binomial Distribution */
  auto cauchit_link_binomial_distrib_two_col = glm(educationX, educationY, 
      new BinomialDistribution!double(), new CauchitLink!double());
  writeln(cauchit_link_binomial_distrib_two_col);
}

