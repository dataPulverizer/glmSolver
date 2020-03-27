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
import glmsolverd.sample;

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
  auto X1 = createRandomMatrix!(double)([20, 5]);
  auto y1 = createRandomColumnVector!(double)(20);
  auto X2 = X1.dup; auto y2 = y1.dup;
  writeln("GELS Test: \n", _gels_test_(X1, y1));
  writeln("GELY Test: \n", _gelss_test_(X2, y2));
  auto invXX = inv(mult_!(double, CblasColMajor, CblasTrans)(X2, X2.dup));
  writeln("Actual (X'X)^(-1): ", invXX);
}

import glmsolverd.io;

/*
  Makes sure that the matrixToBlock() function works
*/
void block_demo()
{
  auto x = createRandomMatrix!(double)([cast(ulong)50, cast(ulong)5]);
  auto xBlock1 = matrixToBlock(x, cast(long)5);
  writeln("x: ", x);
  writeln("xBlock1: ", xBlock1);
  auto xBlock2 = matrixToBlock(x, cast(long)8);
  writeln("xBlock2: ", xBlock2);
}


/*
  Makes sure that eigenvalue decomposition works
*/
void eigen_demo(ulong m)
{
  /* Generate random symmetric matrix */
  auto mat = createSymmetricMatrix!(double, CblasColMajor)(m);
  mat = 0.5*(mat + mat.t());
  auto decomp = eigen(mat);
  writeln("The matrix: ", mat);
  writeln("Eigen values: ", decomp.values);
  writeln("Eigen vectors: ", matrix!(double, CblasColMajor)(decomp.vectors, m));
}

/* Demo createRandomMatrix() */
void random_demo(ulong seed, ulong m)
{
  //auto mat = createRandomMatrix!(double)(m, seed);
  //writeln("Matrix: ", mat);
  auto corr = randomCorrelationMatrix!(double)(m, seed);
  writeln("Matrix: ", corr);
  auto chol = cholesky!('U')(corr);
  writeln("Cholesky decomposition: ", chol);

  //auto X = mvrnorm(20, zerosColumn!(double)(m), corr, seed);
  //writeln("Simulated matrix (X): ", X);
  auto Xy = simulateData!(double, CblasColMajor)(5, 100, 3);
  writeln("Xy: ", Xy);

  /* Sampling from the Poisson Distribution */
  AbstractDistribution!(double) distrib = new PoissonDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto poissSample = _sample_poisson(link.linkinv(Xy.eta), seed);
  writeln("Poisson data: ", poissSample.getData);

  /* Generate GLM data for Poisson distributed Y */
  auto poissonData = simulateData!(double, CblasColMajor)(distrib, 
            link, 5, 50, seed);
  writeln("X: ", poissonData.X);
  writeln("y: ", poissonData.y.getData);

  /* Generate GLM data for binomial distributed Y */
  distrib = new BinomialDistribution!(double)();
  link = new LogitLink!(double)();
  auto binomData = simulateData!(double, CblasColMajor)(distrib, link, 5, 50, seed);
  writeln("Binomial X: ", binomData.X);
  writeln("Binomial y: ", binomData.y.getData);
}


