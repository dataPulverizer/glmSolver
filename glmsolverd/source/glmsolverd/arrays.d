/*
  This module contains implementations for vectors and matrices
*/

module glmsolverd.arrays;

import std.conv: to;
import std.format: format;
import std.traits: isFloatingPoint, isIntegral, isNumeric;
import std.stdio: writeln;
import std.algorithm: min, max;
import std.math: modf;

/********************************************* CBLAS ENUMS *********************************************/
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
      static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
        mixin("return data[dim[0]*j + i] " ~ op ~ "= x;");
      else static assert(0, "Operator \"" ~ op ~ "\" not implemented");
    }
    Matrix!(T, L) opBinary(string op)(Matrix!(T, L) x)
    {
      assert( data.length == x.getData.length,
            "Number of rows and columns in matrices not equal.");
      ulong n = data.length;
      Matrix!(T, L) ret = new Matrix(T, L)(dim[0], dim[1]);
      static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
      {
        for(ulong i = 0; i < n; ++i)
        {
          mixin("ret.getData[i] = " ~ "data[i] " ~ op ~ " x.getData[i];");
        }
      }else static assert(0, "Operator \"" ~ op ~ "\" not implemented");
      return ret;
    }
    void opOpAssign(string op)(Matrix!(T, L) x)
    {
      assert( data.length == x.getData.length,
            "Number of rows and columns in matrices not equal.");
      ulong n = data.length;
      static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
      {
        for(ulong i = 0; i < n; ++i)
        {
          mixin("data[i] " ~ op ~ "= x.getData[i];");
        }
      }else static assert(0, "Operator \"" ~ op ~ "\" not implemented");
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
    @property size() const
    {
      return dim.dup;
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
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
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
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
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
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
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
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
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
    static if((op == "+") | (op == "-") | (op == "*") | (op == "/") | (op == "^^"))
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

/* Aliases */
alias BlockMatrix(T, CBLAS_LAYOUT layout = CblasColMajor) = Matrix!(T, layout)[];
alias BlockVector(T) = Vector!(T)[];
alias BlockColumnVector(T) = ColumnVector!(T)[];
alias BlockRowVector(T) = RowVector!(T)[];

