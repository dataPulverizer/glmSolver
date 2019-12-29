/*
  Linear Algebra Module
*/
module linearalgebra;

import arrays;
import arraycommon;
import apply;
import std.conv: to;
import std.typecons: Tuple, tuple;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

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
  ColumnVector!(T) coef = solve(R, z);
  //auto ret = tuple!("coef", "R")(coef, R);
  //writeln("Coefficient & R:\n", ret);
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

