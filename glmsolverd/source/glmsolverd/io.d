/*
  Module for binary IO
*/
module glmsolverd.io;

import glmsolverd.arrays;
import std.stdio: File;
import std.algorithm.sorting: sort;
import std.array: array;
import std.stdio: writeln;

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

string[] listFiles(string path, bool doSort = true)
{
  auto files = dirEntries(path, SpanMode.breadth);
  string[] output;
  foreach (string fileName; files)
  {
    output ~= fileName;
  }
  if(doSort)
    output = output.sort.array;
  return output;
}

import std.file: dirEntries, SpanMode;
BlockMatrix!(T, layout) readBlockMatrix(T, CBLAS_LAYOUT layout = CblasColMajor)(string path)
{
  BlockMatrix!(T, layout) blockMatrix;
  //auto files = dirEntries(path, SpanMode.breadth);
  string[] files = listFiles(path);
  //writeln("Files: ", files);
  foreach (string fileName; files)
  {
    /* try-catch block here? */
    blockMatrix ~= readMatrix!(T, layout)(fileName);
  }
  return blockMatrix;
}

import std.file: mkdir;
import std.conv : to;
void writeBlockMatrix(T, CBLAS_LAYOUT layout = CblasColMajor)(BlockMatrix!(T, layout) blockMatrix, string path)
{
  mkdir(path); int id = 1;
  foreach(Matrix!(T, layout) mat; blockMatrix)
  {
    /*
    ** try-catch bloc here?
    */
    string fileName = path ~ "/block_" ~ to!(string)(id) ~ ".bin";
    writeMatrix(fileName, mat);
    ++id;
  }
}


