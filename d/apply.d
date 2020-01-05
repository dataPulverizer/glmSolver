/*
  Functions for applying functions over vectors and matrices
*/

module apply;
import arrays;
import arraycommon;

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
  T[] ret = new T[v.length];
  for(int i = 0; i < ret.length; ++i)
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
/* Map a function over a matrix */
Matrix!(T, layout) map(alias fun, T, layout)(Matrix!(T, layout) mat)
{
  auto data = mat.getData;
  auto ret = new T[length(data)];
  for(int i = 0; i < length(ret); ++i)
    ret[i] = fun(data[i]);
  return Matrix!(T, layout)(ret, mat.size);
}

/* This is an inplace map */
Matrix!(T, layout) imap(alias fun, T, CBLAS_LAYOUT layout)(ref Matrix!(T, layout) mat)
{
  auto data = mat.getData;
  for(int i = 0; i < data.length; ++i)
    data[i] = fun(data[i]);
  mat = new Matrix!(T, layout)(data, mat.size);//incase
  return mat;
}


