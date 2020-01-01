/*
  Linear Algebra Module
*/
module linearalgebra;

import arrays;
import arraycommon;
import apply;
import link;
import distributions;
import std.conv: to;
import std.typecons: Tuple, tuple;
import std.traits: isFloatingPoint, isIntegral, isNumeric;
//import std.stdio: writeln;
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
  /* 
    Solver for calculating coefficients and preparing either R or xwx 
     for later calculating covariance matrix
  */
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef);
  /* Covariance calculation happens at the end of the regression function */
  Matrix!(T, layout) cov(Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw);
}
/**************************************** Vanilla Solver ***************************************/
/*
  Vanilla (default) solver:
  1. Calculates regression weights: W()
  2. Solves (X'WX)b = X'Wy for b: solve()
  3. Calculates (X'WX)^(-1): cov()
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
  Matrix!(T, layout) cov(Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    //writeln("Debug Vanilla Solver: cov()");
    return inv(xwx);
  }
}
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
  Matrix!(T, layout) cov(Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    /* I think an optimization is possible here since R is upper triangular */
    xwx = mult_!(T, layout, CblasTrans)(R, R.dup);
    return inv(xwx);
  }
}
/**************************************** GELY Solver ***************************************/
/*
  gely orthogonal solver for linear works for rank deficient matrices
  returns regression coefficient
*/
auto _gely_(T, CBLAS_LAYOUT layout)(Matrix!(T, layout) X, ColumnVector!(T) y)
{
  int m = cast(int)X.nrow;
  int n = cast(int)X.ncol;
  assert(m > n, "Number of rows is less than the number of columns.");
  auto a = X.getData.dup;
  T[] tau = new T[n];
  int lda = layout == CblasColMajor ? m : n;

  int rank = 0; double rcond = 0; auto jpvt = new int[n];
  int ldb = m;
  
  //int info = gels(layout, 'N', m, n, 1, a.ptr, lda, y.getData.ptr, m);
  int info = gelsy(layout, m, n, 1, a.ptr, lda, y.getData.ptr, 
              ldb, jpvt.ptr, rcond, &rank);
  assert(info == 0, "Illegal value info " ~ to!(string)(info) ~ 
                    " from function gelsy");
    
  return new ColumnVector!(T)(y.getData[0..n]);
}
/*
  GELY Solver Using Orthogonal Factorization
*/
class GELYSolver(T, CBLAS_LAYOUT layout = CblasColMajor): AbstractSolver!(T, layout)
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
  void solve(ref Matrix!(T, layout) R, ref Matrix!(T, layout) xwx, 
              ref Matrix!(T, layout) xw, ref Matrix!(T, layout) x,
              ref ColumnVector!(T) z, ref ColumnVector!(T) w, 
              ref ColumnVector!(T) coef)
  {
    xw = sweep!( (x1, x2) => x1 * x2 )(x, w);
    auto zw = map!( (x1, x2) => x1 * x2 )(z, w);
    coef = _gely_!(T, layout)(xw, zw);
    return;
  }
  Matrix!(T, layout) cov(Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    xwx = mult_!(T, layout, CblasTrans)(xw, xw.dup);
    return inv(xwx);
  }
}
/**************************************** GELSS Solver ***************************************/

/* Gets the V matrix from the otput of gelss */
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
  Matrix!(T, layout) cov(Matrix!(T, layout) R, Matrix!(T, layout) xwx, Matrix!(T, layout) xw)
  {
    /* Here R is actually iVt invers transpose of V from SVD of X */
    return mult_!(T, layout, CblasTrans)(R, R.dup);
  }
}

