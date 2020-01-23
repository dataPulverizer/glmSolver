/*
  Convenience functions for vectors and matrices
*/
module glmsolverd.common;

import glmsolverd.arrays;

import std.random;
import std.algorithm: fold;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

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
/* Random number generator for block matrices */
BlockMatrix!(T, layout) createRandomBlockMatrix(T = double, CBLAS_LAYOUT layout = CblasColMajor)(ulong m, ulong n, ulong nBlocks)
{
  BlockMatrix!(T, layout) ret = new Matrix!(T, layout)[nBlocks];
  for(ulong i = 0; i < nBlocks; ++i)
    ret[i] = createRandomMatrix(m, n);
  return ret;
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

/* Unsafe but fast initialization for vector & matrix */
import core.stdc.stdlib: malloc;
/* Initialize array using a pointer */
ColumnVector!(T) fillColumn(T)(T x, ulong n)
{
  T* arr;
  if(n > 0)
  {
    arr = cast(T*)malloc(T.sizeof * n);
    if(arr == null)
      assert(0, "Array Allocation Failed!");
    for(ulong i = 0; i < n; ++i)
      arr[i] = x;
    return new ColumnVector!(T)(arr[0..n]);
  }
  return new ColumnVector!(T)(new T[0]);
}
ColumnVector!(T) zerosColumn(T)(ulong n)
{
  return fillColumn!(T)(cast(T)0, n);
}
/* Initialize array using a pointer */
Matrix!(T, layout) fillMatrix(T, CBLAS_LAYOUT layout = CblasColMajor)(T x, ulong nrow, ulong ncol)
{
  ulong n = nrow * ncol;
  auto arr = cast(T*)malloc(T.sizeof*n);
  if(arr == null)
    assert(0, "Array Allocation Failed!");
  for(ulong i = 0; i < n; ++i)
    arr[i] = x;
  return new Matrix!(T, layout)(arr[0..n], [nrow, ncol]);
}
Matrix!(T, layout) fillMatrix(T, CBLAS_LAYOUT layout = CblasColMajor)(T x, ulong[] dim)
{
  return fillMatrix!(T, layout)(x, dim[0], dim[1]);
}
Matrix!(T, layout) zerosMatrix(T, CBLAS_LAYOUT layout = CblasColMajor)(ulong[] dim)
{
  return fillMatrix!(T, layout)(cast(T)0, dim[0], dim[1]);
}
Matrix!(T, layout) zerosMatrix(T, CBLAS_LAYOUT layout = CblasColMajor)(ulong nrow, ulong ncol)
{
  return fillMatrix!(T, layout)(cast(T)0, nrow, ncol);
}


/*
auto fillColumn(T)(T x, ulong n)
{
  auto v = new T[n];
  for(ulong i = 0; i < n; ++i)
    v[i] = x;
  return new ColumnVector!T(v);
}
auto zerosColumn(T)(ulong n)
{
  return fillColumn!(T)(0, n);
}
*/

auto onesColumn(T)(ulong n)
{
  return fillColumn!(T)(cast(T)1, n);
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
/* Random number generator for block matrices */
BlockColumnVector!(T) createRandomBlockColumnVector(T = double)(ulong m, ulong nBlocks)
{
  BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
  for(ulong i = 0; i < nBlocks; ++i)
    ret[i] = createRandomColumnVector(m);
  return ret;
}

