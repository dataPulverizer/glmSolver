import std.stdio: File, writeln;

/* ldc2 IO.d && ./IO */

/* Functions to read and write 1D arrays */
void write1DArray(T)(string fileName, T[] v)
{
  auto file = File(fileName, "wb");
  size_t[1] n = [v.length];
  file.rawWrite(n);
  file.rawWrite(v);
  return;
}
T[] read1DArray(T)(string fileName)
{
  auto file = File(fileName, "rb");
  size_t[1] n;
  file.rawRead(n);
  auto v = new T[n[0]];
  file.rawRead(v);
  return v;
}

/* Function to read 2D Array */
void write2DArray(T)(string fileName, T[] mat, size_t[] dim)
{
  assert(dim.length == 2, "Array dimensions is not 2.");
  auto file = File(fileName, "wb");
  file.rawWrite(dim);
  file.rawWrite(mat);
  return;
}
T[] read2DArray(T)(string fileName)
{
  auto file = File(fileName, "rb");
  size_t[2] dim;
  file.rawRead(dim);
  auto mat = new T[dim[0]*dim[1]];
  file.rawRead(mat);
  return mat;
}

void main()
{
  double[] x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
  write1DArray("Array.bin", x);
  writeln("Written array: ", x);
  auto y = read1DArray!double("Array.bin");
  writeln("Read array: ", y);
  //auto mat = read2DArray!double("2DRFile.bin");
  //writeln("Reading flattened 2D Array: ", mat);
  //write2DArray("2DDFile.bin", x, [5, 3]);
}
