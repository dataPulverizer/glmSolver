# Enums for BLAS

## Matrix Ordering

**BLAS_ROWMAJOR**
matrix elements are stored in row-major order (default for Perl)

**BLAS_COLMAJOR**
matrix elements are stored in column-major order

## Matrix Operations

**BLAS_NO_TRANS**
operate with the matrix (default)

**BLAS_TRANS**
operate with the transpose matrix

**BLAS_CONJ_TRANS**
operate with the conjugate transpose matrix

## Triangular Matrices
The constants in this group can be imported via the :uplo tag.

**BLAS_UPPER**
refer to upper triangular matrix (default)

**BLAS_LOWER**
refer to lower triangular matrix

The constants in this group can be imported via the :diag tag.

**BLAS_NON_UNIT_DIAG**
non-unit triangular matrix (default)

**BLAS_UNIT_DIAG**
unit triangular matrix, that is diagonal matrix elements are assumed to be one

## Operation Side
The constants in this group can be imported via the :side tag.

**BLAS_LEFT_SIDE**
operate on the left-hand side (default)

**BLAS_RIGHT_SIDE**
operate on the right-hand side

## Vector and Matrix Norms

**BLAS_ONE_NORM**
one-norm (default)

**BLAS_REAL_ONE_NORM**
real one-norm

**BLAS_TWO_NORM**
two-norm

**BLAS_FROBENIUS_NORM**
Frobenius-norm

**BLAS_INF_NORM**
infinity-norm

**BLAS_REAL_INF_NORM**
real infinity-norm

**BLAS_MAX_NORM**
maximum-norm

**BLAS_REAL_MAX_NORM**
real maximum-norm

## Sorting Order

**BLAS_INCREASING_ORDER**
sort in increasing order (default)

**BLAS_DECREASING_ORDER**
sort in decreasing order

## Complex Matrix Elements

**BLAS_NO_CONJ**
operate with the complex vector (default)

**BLAS_CONJ**
operate with the conjugate of the complex vector

## Jacobi Rotations

**BLAS_JROT_INNER**
inner rotation (default)

**BLAS_JROT_OUTER**
outer rotation

**BLAS_JROT_SORTED**
sorted rotation

## Index Base

**BLAS_ZERO_BASE**
indices are zero-based (default for Perl)

**BLAS_ONE_BASE**
indices are one-based

## Symmetric Matrices

**BLAS_GENERAL**
general matrix (default)

**BLAS_SYMMETRIC**
symmetric matrix

**BLAS_HERMITIAN**
Hermitian matrix

**BLAS_TRIANGULAR**
triangular matrix

**BLAS_LOWER_TRIANGULAR**
lower triangular matrix

**BLAS_UPPER_TRIANGULAR**
upper triangular matrix

**BLAS_LOWER_SYMMETRIC**
only the lower half of a symmetric matrix is specified

**BLAS_UPPER_SYMMETRIC**
only the upper half of a symmetric matrix is specified

**BLAS_LOWER_HERMITIAN**
only the lower half of a Hermitian matrix is specified

**BLAS_UPPER_HERMITIAN**
only the upper half of a Hermitian matrix is specified

## BLAS Enum Values

```
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};

enum blas_order {
            blas_rowmajor = 101,
            blas_colmajor = 102 };

enum blas_trans {
            blas_no_trans   = 111,
            blas_trans      = 112,
            blas_conj_trans = 113 };

enum blas_uplo  {
            blas_upper = 121,
            blas_lower = 122 };

enum blas_diag {
            blas_non_unit_diag = 131,
            blas_unit_diag     = 132 };

enum blas_side {
            blas_left_side  = 141,
            blas_right_side = 142 };

enum blas_cmach {
            blas_base      = 151,
            blas_t         = 152,
            blas_rnd       = 153,
            blas_ieee      = 154,
            blas_emin      = 155,
            blas_emax      = 156,
            blas_eps       = 157,
            blas_prec      = 158,
            blas_underflow = 159,
            blas_overflow  = 160,
            blas_sfmin     = 161};

enum blas_norm {
            blas_one_norm       = 171,
            blas_real_one_norm  = 172,
            blas_two_norm       = 173,
            blas_frobenius_norm = 174,
            blas_inf_norm       = 175,
            blas_real_inf_norm  = 176,
            blas_max_norm       = 177,
            blas_real_max_norm  = 178 };

enum blas_sort {
            blas_increasing_order = 181,
            blas_decreasing_order = 182 };

enum blas_conj {
            blas_conj    = 191,
            blas_no_conj = 192 };

enum blas_jrot {
            blas_jrot_inner  = 201,
            blas_jrot_outer  = 202,
            blas_jrot_sorted = 203 };

enum blas_prec {
            blas_prec_single     = 211,
            blas_prec_double     = 212,
            blas_prec_indigenous = 213,
            blas_prec_extra      = 214 };

enum blas_base {
            blas_zero_base = 221,
            blas_one_base  = 222 };

enum blas_symmetry {
            blas_general          = 231,
            blas_symmetric        = 232,
            blas_hermitian        = 233,
            blas_triangular       = 234,
            blas_lower_triangular = 235,
            blas_upper_triangular = 236,
            blas_lower_symmetric  = 237,
            blas_upper_symmetric  = 238,
            blas_lower_hermitian  = 239,
            blas_upper_hermitian  = 240  };

enum blas_field {
            blas_complex          = 241,
            blas_real             = 242,
            blas_double_precision = 243,
            blas_single_precision = 244  };

enum blas_size {
            blas_num_rows      = 251,
            blas_num_cols      = 252,
            blas_num_nonzeros  = 253  };

enum blas_handle{
            blas_invalid_handle = 261,
			blas_new_handle     = 262,
			blas_open_handle    = 263,
			blas_valid_handle   = 264};

enum blas_sparsity_optimization {
            blas_regular       = 271,
            blas_irregular     = 272,
            blas_block         = 273,
            blas_unassembled   = 274 };
```