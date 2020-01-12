module demos.scratch;

import std.conv: to;
import std.stdio : writeln;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;
import glmsolverd.link;
import glmsolverd.distributions;
import glmsolverd.tools;
import glmsolverd.linearalgebra;

import std.stdio : writeln;
import std.typecons: Tuple, tuple;

auto _gels_test_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) X, ColumnVector!(T) y)
{
  int m = cast(int)X.nrow;
  int n = cast(int)X.ncol;
  assert(m > n, "Number of rows is less than the number of columns.");
  auto a = X.getData.dup;
  T[] tau = new T[n];
  int lda = layout == CblasColMajor ? m : n;
  
  int info = gels(layout, 'N', m, n, 1, a.ptr, lda, y.getData.ptr, m);
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function gels");
  
  auto QR = matrix(a, [m, n]);
  auto coef = new ColumnVector!(T)(y.getData[0..n]);
    
  return tuple!("coef", "QR")(coef, QR);
}


auto _gely_test_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) X, ColumnVector!(T) y)
{
  int m = cast(int)X.nrow;
  int n = cast(int)X.ncol;
  assert(m > n, "Number of rows is less than the number of columns.");
  auto a = X.getData.dup;
  T[] tau = new T[n];
  int lda = layout == CblasColMajor ? m : n;

  int rank = 0; double rcond = 0; auto jpvt = new int[n];
  int ldb = m;
  
  int info = gelsy(layout, m, n, 1, a.ptr, lda, y.getData.ptr, 
              ldb, jpvt.ptr, rcond, &rank);
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function gelsy");
  
  auto Orth = matrix(a, [m, n]);
  auto coef = new ColumnVector!(T)(y.getData[0..n]);

  writeln("pivot vector: ", jpvt);
  writeln("conditioning number: ", rcond^^-1);
    
  return tuple!("coef", "Orth")(coef, Orth);
}

auto _gelss_test_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) X, ColumnVector!(T) y)
{
  int m = cast(int)X.nrow;
  int n = cast(int)X.ncol;
  assert(m > n, "Number of rows is less than the number of columns.");
  auto a = X.getData.dup;
  T[] tau = new T[n];
  int lda = layout == CblasColMajor ? m : n;

  int rank = 0; double rcond = 0; int ldb = m; auto s = new T[n];
  
  int info = gelss(layout, m, n, 1, a.ptr, lda, y.getData.ptr,
                ldb, s.ptr, rcond, &rank);
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function gelss");
  
  auto Vt = getV(matrix(a, [m, n]));
  auto coef = new ColumnVector!(T)(y.getData[0..n]);
  /* Inverse of Sigma */
  auto isigma = columnVector(map!( (T x ) => x^^-1 )(s));
  /* Inverse transpose of V */
  auto iVt = sweep!( (x1, x2) => x1 * x2 )(Vt, isigma);
  writeln("conditioning number: ", rcond^^-1);
  writeln("Calculated (X'X)^(-1): ", mult_!(T, layout, CblasTrans)(iVt, iVt.dup));
  
  return tuple!("coef", "iVt")(coef, iVt);
}

/* ldc2 scratch.d arrays.d arraycommon.d apply.d link.d distributions.d tools.d linearalgebra.d io.d fit.d -O2 -L-lopenblas -L-lpthread -L-llapacke -L-llapack -L-lm && ./scratch */
void inverse_test() /* delete this */
{
  auto X1 = createRandomMatrix!(double)(20, 5);
  auto y1 = createRandomColumnVector!(double)(20);
  auto X2 = X1.dup; auto y2 = y1.dup;
  writeln("GELS Test: \n", _gels_test_(X1, y1));
  writeln("GELY Test: \n", _gelss_test_(X2, y2));
  auto invXX = inv(mult_!(double, CblasColMajor, CblasTrans)(X2, X2.dup));
  writeln("Actual (X'X)^(-1): ", invXX);
}
