/*
  Linear Algebra Module
*/
module glmsolverd.linearalgebra;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;
import glmsolverd.link;
import glmsolverd.distributions;
import glmsolverd.tools;

import std.conv: to;
import std.typecons: Tuple, tuple;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

import std.parallelism;
import std.range : iota;
import std.math: pow;

import std.stdio: writeln;
/*
** TODO: Remove the functions that are no longer needed and remove them
** in the demo.d script where they are used in the first visual testing
** phase. The implementation is now very much driven by solver and
** inverse classes.
*/
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
  
  /* See Intel Math Kernel Library documentation for more details */
  /* Routines for Calculating Matrix Inverses */

  /* "General" Matrix Inverse Using LU Decomposition */
  int LAPACKE_sgetrf(int matrix_layout, int m, int n, float * a, int lda, int* ipiv);
  int LAPACKE_dgetrf(int matrix_layout, int m, int n, double* a, int lda, int* ipiv);
  int LAPACKE_sgetri(int matrix_layout, int n, float* a, int lda, in int* ipiv);
  int LAPACKE_dgetri(int matrix_layout, int n, double* a, int lda, in int* ipiv);
  /* Inverse Using Cholesky Decomposition */
  int LAPACKE_spotrf(int matrix_layout, char uplo, int n, float* a, int lda);
  int LAPACKE_dpotrf(int matrix_layout, char uplo, int n, double* a, int lda);
  int LAPACKE_spotri(int matrix_layout, char uplo, int n, float* a, int lda);
  int LAPACKE_dpotri(int matrix_layout, char uplo, int n, double* a, int lda);
  /* Inverse Of Symmetrix Matrix Using LDL Decomposition */
  int LAPACKE_ssytrf(int layout, char uplo, int n, float * a, int lda, int* ipiv);
  int LAPACKE_dsytrf(int layout, char uplo, int n, double * a, int lda, int* ipiv);
  int LAPACKE_ssytri(int layout, char uplo, int n, float* a, int lda, in int* ipiv);
  int LAPACKE_dsytri(int layout, char uplo, int n, double* a, int lda, in int* ipiv);

  /* Functions used in preliminary solver function */
  int LAPACKE_dgetrs(int matrix_layout, char trans, int n, int nrhs, in double* a, int lda, in int* ipiv, double* b, int ldb);
  int LAPACKE_dpotrs(int matrix_layout, char uplo, int n, int nrhs, in double* a, int lda, double* b, int ldb);

  /* Norm of an array */
  double cblas_dnrm2(in int n , in double* x , in int incx);

  int LAPACKE_sgesvd(int matrix_layout, char jobu, char jobvt, int m, int n, float* a, 
                      int lda, float* s, float* u, int ldu, float* vt, int ldvt, float* superb);
  int LAPACKE_dgesvd(int matrix_layout, char jobu, char jobvt, int m, int n, double* a, 
                      int lda, double* s, double* u, int ldu, double* vt, int ldvt, double* superb);
  int LAPACKE_dgeqrf(int matrix_layout, int m, int n, double* a, int lda, double* tau);
  int LAPACKE_dtrtrs(int matrix_layout, char uplo, char trans, char diag, int n, int nrhs, in double* a, int lda , double* b, int ldb);
  int LAPACKE_dorgqr(int matrix_layout, int m, int n, int k, double* a, int lda, in double* tau);
  
  /* Linear Equation solvers */
  int LAPACKE_sgesv(int matrix_layout, int n, int nrhs, float *a, int lda, int* ipiv, float* b, int ldb);
  int LAPACKE_dgesv(int matrix_layout, int n, int nrhs, double *a, int lda, int* ipiv, double* b, int ldb);
  /* Cholesky Decomposition Solver Using (for positive definite matrices) */
  int LAPACKE_sposv(int matrix_layout, char uplo, int n, int nrhs, float * a, int lda, float * b, int ldb);
  int LAPACKE_dposv(int matrix_layout, char uplo, int n, int nrhs, double * a, int lda, double * b, int ldb);
  /* LDL Decomposition Solver (for symmetrix matrices) */
  int LAPACKE_ssysv(int matrix_layout, char uplo, int n, int nrhs, float * a, int lda, int * ipiv, float * b, int ldb);
  int LAPACKE_dsysv(int matrix_layout, char uplo, int n, int nrhs, double * a, int lda, int * ipiv, double * b, int ldb);

  /* Least Squares Solvers */
  /* General Solver Using LU Decomposition */
  int LAPACKE_sgels(int matrix_layout, char trans, int m, int n, int nrhs, float* a, int lda, float* b, int ldb);
  int LAPACKE_dgels(int matrix_layout, char trans, int m, int n, int nrhs, double* a, int lda, double* b, int ldb);
  /* Orthogonal Solver */
  int LAPACKE_sgelsy(int matrix_layout, int m, int n, int nrhs, float* a, int lda, float* b, int ldb, int* jpvt, float rcond, int* rank);
  int LAPACKE_dgelsy(int matrix_layout, int m, int n, int nrhs, double* a, int lda, double* b, int ldb, int* jpvt, double rcond, int* rank);
  /* Minimum Norm SVD Solver */
  int LAPACKE_sgelss(int matrix_layout, int m, int n, int nrhs, float* a, int lda, float* b, int ldb, float* s, float rcond, int* rank);
  int LAPACKE_dgelss(int matrix_layout, int m, int n, int nrhs, double* a, int lda, double* b, int ldb, double* s, double rcond, int* rank);
  /* Minimum Norm SVD Solver Using Divide & Conquer Algorithms */
  int LAPACKE_sgelsd(int matrix_layout, int m, int n, int nrhs, float* a, int lda, float* b, int ldb, float* s, float rcond, int* rank);
  int LAPACKE_dgelsd(int matrix_layout, int m, int n, int nrhs, double* a, int lda, double* b, int ldb, double* s, double rcond, int* rank);

  /* Set the number of threads for blas/lapack in openblas */
  void openblas_set_num_threads(int num_threads);
  //void omp_set_num_threads(int num_threads);
}

alias cblas_dgemm dgemm;
alias cblas_dgemv dgemv;
alias LAPACKE_dgetrf dgetrf;
alias LAPACKE_dgetri dgetri;
alias LAPACKE_dgesvd dgesvd;

/* Singular Value Decomposition */
alias LAPACKE_sgesvd gesvd;
alias LAPACKE_dgesvd gesvd;

alias LAPACKE_dpotrf dpotrf;
alias LAPACKE_dpotri dpotri;
alias LAPACKE_dgetrs dgetrs;
alias LAPACKE_dpotrs dpotrs;
alias LAPACKE_dgeqrf dgeqrf;
alias LAPACKE_dtrtrs dtrtrs;
alias LAPACKE_dorgqr dorgqr;
alias LAPACKE_dgels dgels;

/* Linear Equation Solvers */
/* LU Decomposition Solver */
alias LAPACKE_sgesv gesv;
alias LAPACKE_dgesv gesv;
/* Cholesky Solver */
alias LAPACKE_sposv posv;
alias LAPACKE_dposv posv;
/* LDL Solver */
alias LAPACKE_ssysv sysv;
alias LAPACKE_dsysv sysv;

/* Least Squares Solvers */
/* LU Solver */
alias LAPACKE_dgels gels;
alias LAPACKE_sgels gels;
/* Orthogonal Factorization Solver */
alias LAPACKE_sgelsy gelsy;
alias LAPACKE_dgelsy gelsy;
/* SVD Solver */
alias LAPACKE_sgelss gelss;
alias LAPACKE_dgelss gelss;
/* SVD Solver Using Divide & Conquer */
alias LAPACKE_sgelsd gelsd;
alias LAPACKE_dgelsd gelsd;

/* "General" Matrix Inverse Using LU Decomposition */
alias LAPACKE_sgetrf getrf;
alias LAPACKE_dgetrf getrf;
alias LAPACKE_sgetri getri;
alias LAPACKE_dgetri getri;
/* Inverse Using Cholesky Decomposition */
alias LAPACKE_spotrf potrf;
alias LAPACKE_dpotrf potrf;
alias LAPACKE_spotri potri;
alias LAPACKE_dpotri potri;
/* Inverse Of Symmetrix Matrix Using LDL Decomposition */
alias LAPACKE_ssytrf sytrf;
alias LAPACKE_dsytrf sytrf;
alias LAPACKE_ssytri sytri;
alias LAPACKE_dsytri sytri;

/* Norm function */
double norm(int incr = 1)(double[] x)
{
	return cblas_dnrm2(cast(int)x.length, x.ptr , incr);
}
/********************************************* Matrix Multiplication ******************************************/
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
  m = cast(int)a.nrow;
  n = cast(int)a.ncol;
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
  T[] data = mat.getData.dup;
  
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

alias _solve = solve;

/********************************************* Matrix Inverses ******************************************/
/* Returns the inverse of a matrix */
Matrix!(T, layout) inv(CBLAS_SYMMETRY symmetry = CblasGeneral, T, CBLAS_LAYOUT layout)(Matrix!(T, layout) mat){
	int p = cast(int)mat.nrow;
	int[] ipiv; ipiv.length = p;
  T[] data = mat.getData.dup;
  
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
	double[] a = mat.getData.dup;
  int m = cast(int)mat.nrow;
  int info = 0; 
  auto s = new double[m];
  auto u = new double[m*m];
  auto vt = new double[m*m];
  auto superb = new double[m-1];
  int output = gesvd(CblasColMajor, 'A', 'A', m, m, a.ptr, m, s.ptr, u.ptr, m, vt.ptr, m, superb.ptr );
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
/******************************************* QR Decomposition Functions *********************************/
/*
  Convert outcome from QR algorithm to R Matrix. See page 723 from the 
  MKL library. Table "Computational Routines for Orthogonal Factorization".
  Gets the Upper R matrix from the QR decomposition.
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
  info = dtrtrs(layout, 'U', 'N', 'N', n, 1, R.getData.ptr, n , z.getData.ptr, n);
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function LAPACKE_dtrtrs");
  //ColumnVector!(T) coef = solve(R, z);
  
  //auto ret = tuple!("coef", "R")(coef, R);
  //writeln("Coefficient & R:\n", ret);
  //return tuple!("coef", "R")(coef, R);
  
  return tuple!("coef", "R")(z, R);
}
auto qrls2(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) X, ColumnVector!(T) y)
{
  int m = cast(int)X.nrow;
  int n = cast(int)X.ncol;
  assert(m > n, "Number of rows is less than the number of columns.");
  auto a = X.getData.dup;
  T[] tau = new T[n];
  int lda = layout == CblasColMajor ? m : n;
  int info = gels(layout, 'N', m, n, 1, a.ptr, lda, y.getData.ptr, m);
  //int info = dgeqrf(layout, m, n, a.ptr, lda, tau.ptr);
  
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function LAPACKE_dgels");
  //writeln("tau:", tau);
  auto Q = matrix(a, [m, n]);
  auto R = qrToR(Q);
  auto coef = new ColumnVector!(T)(y.getData[0..n]);
    
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
  The arguments are passed by reference and are updated by the
  QR solver function. It is a "home-brewed" QR solver that uses
  dgeqrf, dorgqr, dtrtrs functions to carry out QR and then
  does an upper triangular solve to return the coefficient
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

/*
  QR Solver Using DGELS function
*/
void _qr_solver2(T, CBLAS_LAYOUT layout = CblasColMajor)
        (ref Matrix!(T, layout) R, ref Matrix!(T, layout) x, 
        ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
        ref ColumnVector!(T) coef)
{
  //writeln("QR Solver");
  auto xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
  auto zw = map!( (x1, x2) => x1 * x2 )(z, w);
  auto coefR = qrls2!(T, layout)(xw, zw);
  coef = coefR.coef;
  R = coefR.R;
}

void _conventional_solver_2(T, CBLAS_LAYOUT layout = CblasColMajor)
        (ref Matrix!(T, layout) xwx, ref Matrix!(T, layout) x,
        ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
        ref ColumnVector!(T) coef)
{
  auto xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
  auto zw = map!( (x1, x2) => x1 * x2 )(z, w);
  xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(xw, xw.dup);
  auto xwz = mult_!(T, layout, CblasTrans)(xw, zw);
  coef = solve(xwx, xwz);
}
/**************************************** Matrix Inverse Classes ***************************************/
/*
  Classes Of Functions For Calculating Matrix Inverses
*/
interface AbstractInverse(T, CBLAS_LAYOUT layout = CblasColMajor)
{
  Matrix!(T, layout) inv(Matrix!(T, layout) mat);
}
/**************************************** GETRI Inverse ***************************************/
/* Calculation Of General Matrix Inverse Using LU Decomposition */
class GETRIInverse(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractInverse!(T, layout)
{
  Matrix!(T, layout) inv(Matrix!(T, layout) A)
  {
    int p = cast(int)A.nrow;
	  auto ipiv = new int[p];
    auto a = A.getData.dup;

    int info = getrf(layout, p, p, a.ptr, p, ipiv.ptr);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function getrf");
    info = getri(layout, p, a.ptr, p, ipiv.ptr);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function getri");
    return matrix!(T, layout)(a, p);
  }
}
/**************************************** POTRI Inverse ***************************************/
/* Calculation Of General Matrix Inverse Using Cholesky Decomposition */
class POTRIInverse(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractInverse!(T, layout)
{
  Matrix!(T, layout) inv(Matrix!(T, layout) A)
  {
    int p = cast(int)A.nrow;
	  auto ipiv = new int[p];
    auto a = A.getData.dup;

    int info = potrf(layout, 'U', p, a.ptr, p);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function potrf");
    info = potri(layout, 'U', p, a.ptr, p);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function potri");
    /* Writing in the lower matrix, dpotri only writes in the upper matrix */
    for(int j = 0; j < p; ++j)
    {
      for(int i = 0; i < j; ++i)
      {
        a[p*i + j] = a[p*j + i];
      }
    }
    return matrix!(T, layout)(a, p);
  }
}
/**************************************** SYTRI Inverse ***************************************/
/* Calculation Of The Inverse Of A Symmetric Matrix Using LDL Decomposition */
class SYTRIInverse(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractInverse!(T, layout)
{
  Matrix!(T, layout) inv(Matrix!(T, layout) A)
  {
    int p = cast(int)A.nrow;
	  auto ipiv = new int[p];
    auto a = A.getData.dup;

    int info = sytrf(layout, 'U', p, a.ptr, p, ipiv.ptr);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function sytrf");
    info = sytri(layout, 'U', p, a.ptr, p, ipiv.ptr);
	  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ " from function sytri");
    /* Writing in the lower matrix, dpotri only writes in the upper matrix */
    for(int j = 0; j < p; ++j)
    {
      for(int i = 0; i < j; ++i)
      {
        a[p*i + j] = a[p*j + i];
      }
    }
    return matrix!(T, layout)(a, p);
  }
}
/**************************************** SVD Inverses ***************************************/
/* Calculation Of The Inverse Of A Symmetric Matrix Using LDL Decomposition */
class SVDInverse(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractInverse!(T, layout)
{
  Matrix!(T, layout) inv(Matrix!(T, layout) A)
  {
    assert(A.nrow == A.ncol, "Number of rows and columns of the matrix are not equal.");
	  auto a = A.getData.dup;
    int m = cast(int)A.nrow;
    int info = 0; 
    auto s = new double[m];
    auto u = new double[m*m];
    auto vt = new double[m*m];
    auto superb = new double[m-1];
    int output = gesvd(CblasColMajor, 'A', 'A', m, m, a.ptr, m, s.ptr, u.ptr, m, vt.ptr, m, superb.ptr);
    assert(info == 0, "gesvd error: U" ~ info.stringof ~ 
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
}
/**************************************** Solver Classes ***************************************/
/*
  Class of functions for calculating coefficients and covariance
*/
interface AbstractSolver(T, CBLAS_LAYOUT layout = CblasColMajor)
{
  /* Calculation for regression weights */
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta);
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              ColumnVector!(T) mu, ColumnVector!(T) eta);
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta);
  BlockColumnVector!(T) W(Block1DParallel dataType, AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta);
  /* 
    Solver for calculating coefficients and preparing either R or xwx 
     for later calculating covariance matrix
  */
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef);
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef);
  void solve(Block1DParallel dataType, ref Matrix!(T, layout) R, 
              ref Matrix!(T, layout) xwx, ref Matrix!(T, layout) xw,
              ref BlockMatrix!(T, layout) x, ref BlockColumnVector!(T) z,
              ref BlockColumnVector!(T) w, ref ColumnVector!(T) coef);
  /* Covariance calculation happens at the end of the regression function */
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, 
                         Matrix!(T, layout) R, Matrix!(T, layout) xwx, 
                         Matrix!(T, layout) xw);
}
/**************************************** Vanilla Solver ***************************************/
/*
  Vanilla (default) solver:
  1. Calculates regression weights: W()
  2. Solves (X'WX)b = X'Wy for b: solve()
  3. Calculates (X'WX)^(-1): cov()
  (Now Obsolete)
*/
class VanillaSolver(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractSolver!(T, layout)
{
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
  {
    return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^(-1);
  }
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
            ColumnVector!T mu, ColumnVector!T eta)
  {
    //writeln("Debug Vanilla Solver: (W)");
    return map!( (T m, T x) => W(distrib, link, m, x) )(mu, eta);
  }
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) W(Block1DParallel dataType, AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    //writeln("Debug Vanilla Solver: solve()");
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(xw, x);
    auto xwz = mult_!(T, layout, CblasTrans)(xw, z);
    coef = _solve(xwx, xwz);
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    //writeln("Debug Vanilla Solver: solve()");
    ulong n = x.length;
    auto blockXW = sweep!( (x1, x2) => x1 * x2 )(x[0], w[0]);
    xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(blockXW, x[0]);
    auto xwz = mult_!(T, layout, CblasTrans)(blockXW, z[0]);
    for(ulong i = 1; i < n; ++i)
    {
      blockXW = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      xwx += mult_!(T, layout, CblasTrans, CblasNoTrans)(blockXW, x[i]);
      xwz += mult_!(T, layout, CblasTrans)(blockXW, z[i]);
    }
    coef = _solve(xwx, xwz);
  }
  void solve(Block1DParallel dataType, ref Matrix!(T, layout) R, 
              ref Matrix!(T, layout) xwx, ref Matrix!(T, layout) xw,
              ref BlockMatrix!(T, layout) x, ref BlockColumnVector!(T) z,
              ref BlockColumnVector!(T) w, ref ColumnVector!(T) coef)
  {
    ulong p = coef.length;
    xwx = zerosMatrix!(T, layout)(p, p);
    ColumnVector!(T) xwz = zerosColumn!(T)(p);

    auto XWX = taskPool.workerLocalStorage(zerosMatrix!(T, layout)(p, p));
    auto XWZ = taskPool.workerLocalStorage(zerosColumn!(T)(p));

    ulong nBlocks = x.length;
    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      auto blockXW = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      XWX.get += mult_!(T, layout, CblasTrans, CblasNoTrans)(blockXW, x[i]);
      XWZ.get += mult_!(T, layout, CblasTrans)(blockXW, z[i]);
    }
    foreach (_xwx; XWX.toRange)
      xwx += _xwx;
    
    foreach (_xwz; XWZ.toRange)
      xwz += _xwz;
    
    coef = _solve(xwx, xwz);
  }
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    return inverse.inv(xwx);
  }
}
/**************************************** Least Squares Solvers ***************************************/
/**************************************** GELS Solver ***************************************/
/*
  gels solver for linear regression returns the coefficients and R the 
  upper triangular R matrix from the QR decomposition
*/
auto _gels_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) X, ColumnVector!(T) y)
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
  //writeln("tau:", tau);
  auto Q = matrix(a, [m, n]);
  auto R = qrToR(Q);
  auto coef = new ColumnVector!(T)(y.getData[0..n]);
    
  return tuple!("coef", "R")(coef, R);
}

/*
  GELS QR Solver:
  1. Calculates regression weights as square root of standard weights: W()
  2. Solves using QR decomposition: solve()
  3. Calculates (R'R)^(-1): cov()
*/
class GELSSolver(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractSolver!(T, layout)
{
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
  {
    return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^-(0.5);
  }
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
            ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!( (T m, T x) => W(distrib, link, m, x) )(mu, eta);
  }
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    auto zw = map!( (x1, x2) => x1 * x2 )(z, w);
    auto coefR = _gels_!(T, layout)(xw, zw);
    coef = coefR.coef;
    R = coefR.R;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    assert(0, "GELSSolver not available for block data.");
  }
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    /* I think an optimization is possible here since R is upper triangular */
    xwx = mult_!(T, layout, CblasTrans)(R, R.dup);
    //return inv(xwx);
    return inverse.inv(xwx);
  }
}
/**************************************** GELY Solver ***************************************/
/*
  gely orthogonal solver for linear works for rank deficient matrices
  returns regression coefficient
*/
auto _gelsy_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) X, ColumnVector!(T) y)
{
  int m = cast(int)X.nrow;
  int n = cast(int)X.ncol;
  assert(m > n, "Number of rows is less than the number of columns.");
  auto a = X.getData.dup;
  T[] tau = new T[n];
  int lda = layout == CblasColMajor ? m : n;

  int rank = 0; T rcond = 0; auto jpvt = new int[n];
  int ldb = m;
  
  //int info = gels(layout, 'N', m, n, 1, a.ptr, lda, y.getData.ptr, m);
  int info = gelsy(layout, m, n, 1, a.ptr, lda, y.getData.ptr, 
              ldb, jpvt.ptr, rcond, &rank);
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function gelsy");
    
  return new ColumnVector!(T)(y.getData[0..n]);
}
/*
  GELSY Solver Using Orthogonal Factorization
*/
class GELSYSolver(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractSolver!(T, layout)
{
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
  {
    return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^-(0.5);
  }
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
            ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!( (T m, T x) => W(distrib, link, m, x) )(mu, eta);
  }
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    auto zw = map!( (x1, x2) => x1 * x2 )(z, w);
    coef = _gelsy_!(T, layout)(xw, zw);
    return;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    assert(0, "GELSSolver not available for block data.");
  }
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    xwx = mult_!(T, layout, CblasTrans)(xw, xw.dup);
    //return inv(xwx);
    return inverse.inv(xwx);
  }
}
/**************************************** GELSS Solver ***************************************/

/* Gets the V matrix from the output of gelss */
auto getV(T, CBLAS_LAYOUT layout = CblasColMajor)(Matrix!(T, layout) a)
{
  ulong p = a.ncol;
  ulong n = p * p;
  auto V = matrix(new T[n], p);
  for(ulong i = 0; i < p; ++i)
  {
    for(ulong j = 0; j < p; ++j)
    {
      V[i, j] = a[i, j];
    }
  }
  return V;
}
/*
  Least Squares SVD Solver
*/
auto _gelss_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) X, ColumnVector!(T) y)
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
  
  return tuple!("coef", "R")(coef, iVt);
}
/* GELSS Solver Using SVD Solver */
class GELSSSolver(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractSolver!(T, layout)
{
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
  {
    return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^-(0.5);
  }
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
            ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!( (T m, T x) => W(distrib, link, m, x) )(mu, eta);
  }
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    auto zw = map!( (x1, x2) => x1 * x2 )(z, w);
    auto coefR = _gelss_!(T, layout)(xw, zw);
    coef = coefR.coef;
    R = coefR.R;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    assert(0, "GELSSolver not available for block data.");
  }
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    /* Here R is actually iVt invers transpose of V from SVD of X */
    return mult_!(T, layout, CblasTrans)(R, R.dup);
  }
}
/**************************************** GELSD Solver ***************************************/
/*
  Least Squares SVD Solver as with gelss but uses divide and conquer
*/
auto _gelsd_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) X, ColumnVector!(T) y)
{
  int m = cast(int)X.nrow;
  int n = cast(int)X.ncol;
  assert(m > n, "Number of rows is less than the number of columns.");
  auto a = X.getData.dup;
  T[] tau = new T[n];
  int lda = layout == CblasColMajor ? m : n;

  int rank = 0; double rcond = 0; int ldb = m; auto s = new T[n];
  
  int info = gelsd(layout, m, n, 1, a.ptr, lda, y.getData.ptr,
                ldb, s.ptr, rcond, &rank);
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function gelsd");
  /* Returns the coefficients */
  return new ColumnVector!(T)(y.getData[0..n]);
}
/* GELSD Divide and Conquer SVD Solver */
class GELSDSolver(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractSolver!(T, layout)
{
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
  {
    return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^-(0.5);
  }
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
            ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!( (T m, T x) => W(distrib, link, m, x) )(mu, eta);
  }
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    auto zw = map!( (x1, x2) => x1 * x2 )(z, w);
    coef = _gelsd_!(T, layout)(xw, zw);
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    assert(0, "GELSSolver not available for block data.");
  }
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    xwx = mult_!(T, layout, CblasTrans)(xw, xw.dup);
    return inverse.inv(xwx);
  }
}
/**************************************** Linear Equation Solvers ***************************************/
/**************************************** GESV Solver ***************************************/
/*
  GESV Solver Using LU Decomposition:
  1. Calculates regression weights: W()
  2. Solves (X'WX)b = X'Wy for b: solve()
  3. Calculates (X'WX)^(-1): cov()
*/
auto _gesv_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) A, ColumnVector!(T) b)
{
  int m = cast(int)A.nrow;
  int n = cast(int)A.ncol;
  assert(m == n, "Matrix must be square.");
  auto a = A.getData.dup; int nrhs = 1;
  int lda = n; int ldb = n; auto ipiv = new int[n];
  
  int info = gesv(layout, n, nrhs, a.ptr, lda, ipiv.ptr, b.getData.ptr, ldb);
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function gesv");
  /* Returns b which is overwritten by coefficients */
  return b;
}

class GESVSolver(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractSolver!(T, layout)
{
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
  {
    return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^(-1);
  }
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
            ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!( (T m, T x) => W(distrib, link, m, x) )(mu, eta);
  }
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) W(Block1DParallel dataType, AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(xw, x);
    auto xwz = mult_!(T, layout, CblasTrans)(xw, z);
    coef = _gesv_(xwx, xwz);
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    ulong p = coef.length;
    xwx = zerosMatrix!(T, layout)(p, p);
    ColumnVector!(T) XWZ = zerosColumn!(T)(p);
    ulong nBlocks = x.length;
    for(ulong i = 0; i < nBlocks; ++i)
    {
      auto tmp = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      xwx += mult_!(T, layout, CblasTrans, CblasNoTrans)( tmp , x[i]);
      XWZ += mult_!(T, layout, CblasTrans)(tmp, z[i]);
    }
    coef = _gesv_(xwx, XWZ);
  }
  void solve(Block1DParallel dataType, ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    ulong p = coef.length;
    xwx = zerosMatrix!(T, layout)(p, p);
    ColumnVector!(T) xwz = zerosColumn!(T)(p);

    auto XWX = taskPool.workerLocalStorage(zerosMatrix!(T, layout)(p, p));
    auto XWZ = taskPool.workerLocalStorage(zerosColumn!(T)(p));

    ulong nBlocks = x.length;
    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      auto tmp = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      XWX.get += mult_!(T, layout, CblasTrans, CblasNoTrans)(tmp , x[i]);
      XWZ.get += mult_!(T, layout, CblasTrans)(tmp, z[i]);
    }
    foreach(_xwx; XWX.toRange)
      xwx += _xwx;
    
    foreach(_xwz; XWZ.toRange)
      xwz += _xwz;
    
    coef = _gesv_(xwx, xwz);
  }
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    return inverse.inv(xwx);
  }
}
/**************************************** POSV Solver writeln ***************************************/
/*
  Cholesky Decomposition Solver For Positive Definite Matrices
*/
auto _posv_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) A, ColumnVector!(T) b)
{
  int m = cast(int)A.nrow;
  int n = cast(int)A.ncol;
  assert(m == n, "Matrix must be square.");
  auto a = A.getData.dup; int nrhs = 1; int lda = n;
  int ldb = n;
  
  int info = posv(layout, 'U', n, nrhs, a.ptr, lda, b.getData.ptr, ldb);
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function posv");
  /* Returns b which is overwritten by coefficients */
  return b;
}

class POSVSolver(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractSolver!(T, layout)
{
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
  {
    return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^(-1);
  }
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
            ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!( (T m, T x) => W(distrib, link, m, x) )(mu, eta);
  }
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) W(Block1DParallel dataType, AbstractDistribution!T distrib,
              AbstractLink!T link, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(xw, x);
    auto xwz = mult_!(T, layout, CblasTrans)(xw, z);
    coef = _posv_(xwx, xwz);
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    ulong p = coef.length;
    xwx = zerosMatrix!(T, layout)(p, p);
    ColumnVector!(T) XWZ = zerosColumn!(T)(p);
    ulong nBlocks = x.length;
    for(ulong i = 0; i < nBlocks; ++i)
    {
      auto tmp = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      xwx += mult_!(T, layout, CblasTrans, CblasNoTrans)( tmp , x[i]);
      XWZ += mult_!(T, layout, CblasTrans)(tmp, z[i]);
    }
    coef = _posv_(xwx, XWZ);
  }
  void solve(Block1DParallel dataType, ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x, ref BlockColumnVector!(T) z,
              ref BlockColumnVector!(T) w, ref ColumnVector!(T) coef)
  {
    ulong p = coef.length;

    xwx = zerosMatrix!(T, layout)(p, p);
    ColumnVector!(T) xwz = zerosColumn!(T)(p);

    auto XWX = taskPool.workerLocalStorage(zerosMatrix!(T, layout)(p, p));
    auto XWZ = taskPool.workerLocalStorage(zerosColumn!(T)(p));
    
    ulong nBlocks = x.length;
    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      auto tmp = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      XWX.get += mult_!(T, layout, CblasTrans, CblasNoTrans)( tmp , x[i]);
      XWZ.get += mult_!(T, layout, CblasTrans)(tmp, z[i]);
    }
    foreach (_xwx; XWX.toRange)
      xwx += _xwx;
    
    foreach (_xwz; XWZ.toRange)
      xwz += _xwz;
    
    coef = _posv_(xwx, xwz);
  }
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    return inverse.inv(xwx);
  }
}
/**************************************** SYSV Solver ***************************************/
/*
  LDL Decomposition Solver For Symmetric Indefinite Matrices
*/
auto _sysv_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) A, ColumnVector!(T) b)
{
  int m = cast(int)A.nrow;
  int n = cast(int)A.ncol;
  assert(m == n, "Matrix must be square.");
  auto a = A.getData.dup; int nrhs = 1; int lda = n;
  int ldb = n; auto ipiv = new int[n];
  
  int info = sysv(layout, 'U', n, nrhs, a.ptr, lda, ipiv.ptr, b.getData.ptr, ldb);
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function sysv");
  /* Returns b which is overwritten by coefficients */
  return b;
}

class SYSVSolver(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractSolver!(T, layout)
{
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
  {
    return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^(-1);
  }
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
            ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!( (T m, T x) => W(distrib, link, m, x) )(mu, eta);
  }
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) W(Block1DParallel dataType, AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(xw, x);
    auto xwz = mult_!(T, layout, CblasTrans)(xw, z);
    coef = _sysv_(xwx, xwz);
  }
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    ulong p = coef.length;
    xwx = zerosMatrix!(T, layout)(p, p);
    ColumnVector!(T) XWZ = zerosColumn!(T)(p);
    ulong nBlocks = x.length;
    for(ulong i = 0; i < nBlocks; ++i)
    {
      auto tmp = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      xwx += mult_!(T, layout, CblasTrans, CblasNoTrans)(tmp , x[i]);
      XWZ += mult_!(T, layout, CblasTrans)(tmp, z[i]);
    }
    coef = _sysv_(xwx, XWZ);
  }
  void solve(Block1DParallel dataType, ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    ulong p = coef.length;

    xwx = zerosMatrix!(T, layout)(p, p);
    ColumnVector!(T) xwz = zerosColumn!(T)(p);

    auto XWX = taskPool.workerLocalStorage(zerosMatrix!(T, layout)(p, p));
    auto XWZ = taskPool.workerLocalStorage(zerosColumn!(T)(p));

    ulong nBlocks = x.length;
    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      auto tmp = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      XWX.get += mult_!(T, layout, CblasTrans, CblasNoTrans)(tmp , x[i]);
      XWZ.get += mult_!(T, layout, CblasTrans)(tmp, z[i]);
    }
    foreach (_xwx; XWX.toRange)
      xwx += _xwx;
    
    foreach (_xwz; XWZ.toRange)
      xwz += _xwz;
    
    coef = _sysv_(xwx, xwz);
  }
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    return inverse.inv(xwx);
  }
}

/**************************************** Gradient Descent Solvers ***************************************/

interface AbstractGradientSolver(T, CBLAS_LAYOUT layout = CblasColMajor)
{
  private:
  ColumnVector!(T) pgradient_(AbstractDistribution!T distrib, AbstractLink!T link,
      ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu, 
      ColumnVector!T eta);

  public:
  /* Weights */
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta);
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
            ColumnVector!T mu, ColumnVector!T eta);
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta);
  BlockColumnVector!(T) W(Block1DParallel dataType, AbstractDistribution!T distrib,
              AbstractLink!T link, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta);

  /* Gradients */

  /* Returns the average gradient calculated using (n - p) */
  ColumnVector!(T) pgradient(AbstractDistribution!T distrib, AbstractLink!T link,
              ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu,
              ColumnVector!T eta);
  ColumnVector!(T) pgradient(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) y, BlockMatrix!(T, layout) x,
              BlockColumnVector!(T) mu, BlockColumnVector!T eta);
  ColumnVector!(T) pgradient(Block1DParallel dataType, AbstractDistribution!T distrib,
              AbstractLink!T link, BlockColumnVector!(T) y, 
              BlockMatrix!(T, layout) x, BlockColumnVector!(T) mu,
              BlockColumnVector!T eta);

  /* Solver for standard matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link,
            ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu,
            ColumnVector!T eta, ref ColumnVector!(T) coef);
  /* Solver for blocked matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link, 
            BlockColumnVector!(T) y, BlockMatrix!(T, layout) x, 
            BlockColumnVector!(T) mu, BlockColumnVector!(T) eta,
            ref ColumnVector!(T) coef);
  /* Solver for parallel blocked matrices/vectors */
  void solve(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!(T) mu,
            BlockColumnVector!(T) eta, ref ColumnVector!(T) coef);
  void XWX(ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w);
  void XWX(ref Matrix!(T, layout) xwx, 
              ref BlockMatrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w);
  void XWX(Block1DParallel dataType, ref Matrix!(T, layout) xwx, 
              ref BlockMatrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w);
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, Matrix!(T, layout) xwx);
  immutable(string) name();
  /* Modify/Unmodify coefficients for Nesterov */
  void NesterovModifier(ref ColumnVector!(T) coef);
  void NesterovUnModifier(ref ColumnVector!(T) coef);
  /* Passes the iteration to the object */
  void passIteration(ulong _iter);
}

/* Gradient Mixin */
mixin template GradientMixin(T, CBLAS_LAYOUT layout)
{
  private:
  ColumnVector!(T) pgradient_(AbstractDistribution!T distrib, AbstractLink!T link,
      ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu, 
      ColumnVector!T eta)
  {
    ulong p = x.ncol;
    ulong ni = x.nrow;
    auto grad = zerosColumn!T(p);
    auto tmp = zerosColumn!(T)(ni);
    for(ulong i = 0; i < ni; ++i)
    {
      tmp[i] = (y[i] - mu[i])/(link.deta_dmu(mu[i], eta[i]) * distrib.variance(mu[i]));
      for(ulong j = 0; j < p; ++j)
      {
        grad[j] += tmp[i] * x[i, j];
      }
    }
    return grad;
  }

  public:
  /* Gradients */
  /* Returns the average gradient calculated using (n - p) */
  ColumnVector!(T) pgradient(AbstractDistribution!T distrib, AbstractLink!T link,
              ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu,
              ColumnVector!T eta)
  {
    ulong p = x.ncol;
    ulong n = x.nrow;
    auto grad = pgradient_(distrib, link, y, x, mu, eta);
    assert(n > p, "Number of items n is not greater than the number of parameters p.");
    return grad/cast(T)(n - p);
  }
  ColumnVector!(T) pgradient(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) y, BlockMatrix!(T, layout) x,
              BlockColumnVector!(T) mu, BlockColumnVector!T eta)
  {
    ulong nBlocks = y.length; ulong n = 0;
    ulong p = x[0].ncol;
    auto grad = zerosColumn!T(p);
    for(ulong i = 0; i < nBlocks; ++i)
    {
      n += y[i].length;
      grad += pgradient_(distrib, link, y[i], x[i], mu[i], eta[i]);
    }
    assert(n > p, "Number of items n is not greater than the number of parameters p.");
    return grad/cast(T)(n - p);
  }
  ColumnVector!(T) pgradient(Block1DParallel dataType, AbstractDistribution!T distrib,
              AbstractLink!T link, BlockColumnVector!(T) y, 
              BlockMatrix!(T, layout) x, BlockColumnVector!(T) mu,
              BlockColumnVector!T eta)
  {
    ulong nBlocks = y.length;
    ulong p = x[0].ncol;

    auto nStore = taskPool.workerLocalStorage(cast(ulong)0);
    auto gradStore = taskPool.workerLocalStorage(zerosColumn!T(p));

    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      nStore.get += y[i].length;
      gradStore.get += pgradient_(distrib, link, y[i], x[i], mu[i], eta[i]);
    }

    ulong n = 0;
    auto grad = zerosColumn!T(p);
    foreach(_n; nStore.toRange)
      n += _n;
    foreach(_grad; gradStore.toRange)
      grad += _grad;
    
    assert(n > p, "Number of items n is not greater than the number of parameters p.");
    return grad/cast(T)(n - p);
  }
  /* Weights */
  T W(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
  {
    return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^(-1);
  }
  ColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
            ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!( (T m, T x) => W(distrib, link, m, x) )(mu, eta);
  }
  BlockColumnVector!(T) W(AbstractDistribution!T distrib, AbstractLink!T link, 
              BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) W(Block1DParallel dataType, AbstractDistribution!T distrib,
              AbstractLink!T link, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = W(distrib, link, mu[i], eta[i]);
    return ret;
  }
  void XWX(ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    xwx = mult_!(T, layout, CblasTrans, CblasNoTrans)(xw, x);
  }
  void XWX(ref Matrix!(T, layout) xwx, 
              ref BlockMatrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w)
  {
    ulong p = x[0].ncol;
    xwx = zerosMatrix!(T, layout)(p, p);
    ulong nBlocks = x.length;
    for(ulong i = 0; i < nBlocks; ++i)
    {
      auto tmp = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      xwx += mult_!(T, layout, CblasTrans, CblasNoTrans)(tmp , x[i]);
    }
  }
  void XWX(Block1DParallel dataType, ref Matrix!(T, layout) xwx, 
              ref BlockMatrix!(T, layout) xw, ref BlockMatrix!(T, layout) x,
              ref BlockColumnVector!(T) z, ref BlockColumnVector!(T) w)
  {
    ulong p = x[0].ncol;

    xwx = zerosMatrix!(T, layout)(p, p);
    auto XWX = taskPool.workerLocalStorage(zerosMatrix!(T, layout)(p, p));

    ulong nBlocks = x.length;
    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      auto tmp = sweep!( (x1, x2) => x1 * x2 )(x[i], w[i]);
      XWX.get += mult_!(T, layout, CblasTrans, CblasNoTrans)(tmp , x[i]);
    }
    foreach (_xwx; XWX.toRange)
      xwx += _xwx;
  }
  Matrix!(T, layout) cov(AbstractInverse!(T, layout) inverse, Matrix!(T, layout) xwx)
  {
    return inverse.inv(xwx);
  }
}


/* Basic Gradient Descent Solver */
/*
  Reference is the website and article by Sebastian Ruder:
  An overview of gradient descent optimization algorithms
  Website: https://ruder.io/optimizing-gradient-descent/index.html
  Article (Arxiv): https://arxiv.org/pdf/1609.04747.pdf
*/
mixin template GradientSolverMixin(T, CBLAS_LAYOUT layout)
{
  private:
  immutable(string) solverName;
  T learningRate;
  public:
  /* Solver for standard matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link,
            ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu,
            ColumnVector!T eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    coef += learningRate * grad;
  }
  /* Solver for blocked matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link, 
            BlockColumnVector!(T) y, BlockMatrix!(T, layout) x, 
            BlockColumnVector!(T) mu, BlockColumnVector!(T) eta,
            ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    coef += learningRate * grad;
  }
  /* Solver for parallel blocked matrices/vectors */
  void solve(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!(T) mu,
            BlockColumnVector!(T) eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(dataType, distrib, link, y, x, mu, eta);
    coef += learningRate * grad;
  }
  immutable(string) name()
  {
    return solverName;
  }
  /* Modify/Unmodify coefficients for Nesterov */
  void NesterovModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void NesterovUnModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void passIteration(ulong _iter){}
  this(T _learningRate)
  {
    learningRate = _learningRate;
    solverName = "Simple Gradient Descent";
  }
}


class GradientDescent(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractGradientSolver!(T, layout)
{
  mixin GradientMixin!(T, layout);
  mixin GradientSolverMixin!(T, layout);
}

/***************** Momentum Gradient Descent Solver **************/
mixin template MomentumMixin(T, CBLAS_LAYOUT layout)
{
  private:
  immutable(string) solverName;
  T learningRate;
  T momentum;
  ColumnVector!(T) delta;
  public:
  /* Solver for standard matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link,
            ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu,
            ColumnVector!T eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    delta = momentum*delta + learningRate*grad;
    coef += delta;
  }
  /* Solver for blocked matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link, 
            BlockColumnVector!(T) y, BlockMatrix!(T, layout) x, 
            BlockColumnVector!(T) mu, BlockColumnVector!(T) eta,
            ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    delta = momentum*delta + learningRate*grad;
    coef += delta;
  }
  /* Solver for parallel blocked matrices/vectors */
  void solve(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!(T) mu,
            BlockColumnVector!(T) eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(dataType, distrib, link, y, x, mu, eta);
    delta = momentum*delta + learningRate*grad;
    coef += delta;
  }
  immutable(string) name()
  {
    return solverName;
  }
  /* Modify/Unmodify coefficients for Nesterov */
  void NesterovModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void NesterovUnModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void passIteration(ulong _iter){}
  this(T _learningRate, T _momentum, ulong p)
  {
    learningRate = _learningRate;
    momentum = _momentum;
    delta = zerosColumn!T(p);
    solverName = "Momentum";
  }
}


class Momentum(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractGradientSolver!(T, layout)
{
  mixin GradientMixin!(T, layout);
  mixin MomentumMixin!(T, layout);
}

/***************** Nesterov Gradient Descent Solver **************/
mixin template NesterovMixin(T, CBLAS_LAYOUT layout)
{
  private:
  immutable(string) solverName;
  T learningRate;
  T momentum;
  ColumnVector!(T) delta;
  public:
  /* Solver for standard matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link,
            ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu,
            ColumnVector!T eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    delta = momentum*delta + learningRate*grad;
    coef += delta;
  }
  /* Solver for blocked matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link, 
            BlockColumnVector!(T) y, BlockMatrix!(T, layout) x, 
            BlockColumnVector!(T) mu, BlockColumnVector!(T) eta,
            ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    delta = momentum*delta + learningRate*grad;
    coef += delta;
  }
  /* Solver for parallel blocked matrices/vectors */
  void solve(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!(T) mu,
            BlockColumnVector!(T) eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(dataType, distrib, link, y, x, mu, eta);
    delta = momentum*delta + learningRate*grad;
    coef += delta;
  }
  immutable(string) name()
  {
    return solverName;
  }
  /* Modify/Unmodify coefficients for Nesterov */
  void NesterovModifier(ref ColumnVector!(T) coef)
  {
    coef -= momentum * delta;
    return;
  }
  void NesterovUnModifier(ref ColumnVector!(T) coef)
  {
    coef += momentum * delta;
    return;
  }
  void passIteration(ulong _iter){}
  this(T _learningRate, T _momentum, ulong p)
  {
    learningRate = _learningRate;
    momentum = _momentum;
    delta = zerosColumn!T(p);
    solverName = "Nesterov";
  }
}


class Nesterov(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractGradientSolver!(T, layout)
{
  mixin GradientMixin!(T, layout);
  mixin NesterovMixin!(T, layout);
}

/***************** Adagrad Gradient Descent Solver **************/
mixin template AdagradMixin(T, CBLAS_LAYOUT layout)
{
  private:
  immutable(string) solverName;
  T learningRate;
  ColumnVector!(T) G;
  T epsilon;

  public:
  /* Solver for standard matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link,
            ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu,
            ColumnVector!T eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    G += (coef^^2);
    coef += (learningRate*grad)/((G + epsilon)^^0.5);
  }
  /* Solver for blocked matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link, 
            BlockColumnVector!(T) y, BlockMatrix!(T, layout) x, 
            BlockColumnVector!(T) mu, BlockColumnVector!(T) eta,
            ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    G += (coef^^2);
    coef += (learningRate*grad)/((G + epsilon)^^0.5);
  }
  /* Solver for parallel blocked matrices/vectors */
  void solve(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!(T) mu,
            BlockColumnVector!(T) eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(dataType, distrib, link, y, x, mu, eta);
    G += (coef^^2);
    coef += (learningRate*grad)/((G + epsilon)^^0.5);
  }
  immutable(string) name()
  {
    return solverName;
  }
  /* Modify/Unmodify coefficients for Nesterov */
  void NesterovModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void NesterovUnModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void passIteration(ulong _iter){}
  this(T _learningRate, T _epsilon, ulong p)
  {
    learningRate = _learningRate;
    epsilon = _epsilon;
    G = zerosColumn!T(p);
    solverName = "Adagrad";
  }
}

class Adagrad(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractGradientSolver!(T, layout)
{
  mixin GradientMixin!(T, layout);
  mixin AdagradMixin!(T, layout);
}

/***************** Adadelta Gradient Descent Solver **************/
mixin template AdadeltaMixin(T, CBLAS_LAYOUT layout)
{
  private:
  immutable(string) solverName;
  T momentum;
  ColumnVector!(T) G;
  ColumnVector!(T) B;
  T epsilon;

  public:
  /* Solver for standard matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link,
            ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu,
            ColumnVector!T eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    auto momentumM1 = (1 - momentum);
    G = (momentum*G) + (momentumM1*(grad^^2));
    ColumnVector!(T) diff = (grad*((B + epsilon)^^0.5))/((G + epsilon)^^0.5);
    coef += diff;
    B += (momentum*B) + (momentumM1*(diff^^2));
  }
  /* Solver for blocked matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link, 
            BlockColumnVector!(T) y, BlockMatrix!(T, layout) x, 
            BlockColumnVector!(T) mu, BlockColumnVector!(T) eta,
            ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    auto momentumM1 = (1 - momentum);
    G = (momentum*G) + (momentumM1*(grad^^2));
    ColumnVector!(T) diff = (grad*((B + epsilon)^^0.5))/((G + epsilon)^^0.5);
    coef += diff;
    B += (momentum*B) + (momentumM1*(diff^^2));
  }
  /* Solver for parallel blocked matrices/vectors */
  void solve(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!(T) mu,
            BlockColumnVector!(T) eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(dataType, distrib, link, y, x, mu, eta);
    auto momentumM1 = (1 - momentum);
    G = (momentum*G) + (momentumM1*(grad^^2));
    ColumnVector!(T) diff = (grad*((B + epsilon)^^0.5))/((G + epsilon)^^0.5);
    coef += diff;
    B += (momentum*B) + (momentumM1*(diff^^2));
  }
  immutable(string) name()
  {
    return solverName;
  }
  /* Modify/Unmodify coefficients for Nesterov */
  void NesterovModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void NesterovUnModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void passIteration(ulong _iter){}
  this(T _momentum, T _epsilon, ulong p)
  {
    momentum = _momentum;
    G = zerosColumn!T(p);
    B = zerosColumn!T(p);
    epsilon = _epsilon;
    solverName = "Adadelta";
  }
}

class Adadelta(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractGradientSolver!(T, layout)
{
  mixin GradientMixin!(T, layout);
  mixin AdadeltaMixin!(T, layout);
}

/***************** RMSprop Gradient Descent Solver **************/
mixin template RMSpropMixin(T, CBLAS_LAYOUT layout)
{
  private:
  immutable(string) solverName;
  T learningRate;
  ColumnVector!(T) G;
  T epsilon;

  public:
  /* Solver for standard matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link,
            ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu,
            ColumnVector!T eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    G = (0.9*G) + (0.1*(grad^^2));
    ColumnVector!(T) diff = (learningRate*grad)/((G + epsilon)^^0.5);
    coef += diff;
  }
  /* Solver for blocked matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link, 
            BlockColumnVector!(T) y, BlockMatrix!(T, layout) x, 
            BlockColumnVector!(T) mu, BlockColumnVector!(T) eta,
            ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    G = (0.9*G) + (0.1*(grad^^2));
    ColumnVector!(T) diff = (learningRate*grad)/((G + epsilon)^^0.5);
    coef += diff;
  }
  /* Solver for parallel blocked matrices/vectors */
  void solve(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!(T) mu,
            BlockColumnVector!(T) eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(dataType, distrib, link, y, x, mu, eta);
    G = (0.9*G) + (0.1*(grad^^2));
    ColumnVector!(T) diff = (learningRate*grad)/((G + epsilon)^^0.5);
    coef += diff;
  }
  immutable(string) name()
  {
    return solverName;
  }
  /* Modify/Unmodify coefficients for Nesterov */
  void NesterovModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void NesterovUnModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void passIteration(ulong _iter){};
  this(T _learningRate, T _epsilon, ulong p)
  {
    learningRate = _learningRate;
    G = zerosColumn!T(p);
    epsilon = _epsilon;
    solverName = "RMSprop";
  }
}


class RMSprop(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractGradientSolver!(T, layout)
{
  mixin GradientMixin!(T, layout);
  mixin RMSpropMixin!(T, layout);
}


/***************** Adam Gradient Descent Solver **************/
mixin template AdamMixin(T, CBLAS_LAYOUT layout)
{
  private:
  immutable(string) solverName;
  T learningRate;
  T b1;
  T b2;
  ColumnVector!(T) m;
  ColumnVector!(T) v;
  ulong iter;
  T epsilon;

  public:
  /* Solver for standard matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link,
            ColumnVector!T y, Matrix!(T, layout) x, ColumnVector!T mu,
            ColumnVector!T eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    m = (b1*m) + ((1 - b1)*grad);
    v = (b2*v) + ((1 - b2)*(grad^^2));
    ColumnVector!(T) mp = m/(1 - (b1^^iter));
    ColumnVector!(T) vp = v/(1 - (b2^^iter));
    ColumnVector!(T) diff = (learningRate * mp)/(epsilon + (vp^^0.5));
    coef += diff;
  }
  /* Solver for blocked matrices/vectors */
  void solve(AbstractDistribution!T distrib, AbstractLink!T link, 
            BlockColumnVector!(T) y, BlockMatrix!(T, layout) x, 
            BlockColumnVector!(T) mu, BlockColumnVector!(T) eta,
            ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(distrib, link, y, x, mu, eta);
    m = (b1*m) + ((1 - b1)*grad);
    v = (b2*v) + ((1 - b2)*(grad^^2));
    ColumnVector!(T) mp = m/(1 - (b1^^iter));
    ColumnVector!(T) vp = v/(1 - (b2^^iter));
    ColumnVector!(T) diff = (learningRate * mp)/(epsilon + (vp^^0.5));
    coef += diff;
  }
  /* Solver for parallel blocked matrices/vectors */
  void solve(Block1DParallel dataType, AbstractDistribution!T distrib, 
            AbstractLink!T link, BlockColumnVector!(T) y, 
            BlockMatrix!(T, layout) x, BlockColumnVector!(T) mu,
            BlockColumnVector!(T) eta, ref ColumnVector!(T) coef)
  {
    auto grad = pgradient(dataType, distrib, link, y, x, mu, eta);
    m = (b1*m) + ((1 - b1)*grad);
    v = (b2*v) + ((1 - b2)*(grad^^2));
    ColumnVector!(T) mp = m/(1 - (b1^^iter));
    ColumnVector!(T) vp = v/(1 - (b2^^iter));
    ColumnVector!(T) diff = (learningRate * mp)/(epsilon + (vp^^0.5));
    coef += diff;
  }
  immutable(string) name()
  {
    return solverName;
  }
  /* Modify/Unmodify coefficients for Nesterov */
  void NesterovModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void NesterovUnModifier(ref ColumnVector!(T) coef)
  {
    return;
  }
  void passIteration(ulong _iter)
  {
    iter = _iter;
  }
  /* b1 = 0.9, b2 = 0.999 */
  this(T _learningRate, T _b1, T _b2, T _epsilon, ulong p)
  {
    learningRate = _learningRate;
    epsilon = _epsilon; b1 = _b1; b2 = _b2;
    m = zerosColumn!T(p);
    v = zerosColumn!T(p);
    iter = 1; solverName = "Adam";
  }
}

class Adam(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractGradientSolver!(T, layout)
{
  mixin GradientMixin!(T, layout);
  mixin AdamMixin!(T, layout);
}


