/*
  Module for binary IO
*/
module glmsolverd.io;

import glmsolverd.arrays;
import std.stdio: File;

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

