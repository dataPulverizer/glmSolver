# GLMSolver

Solvers for GLM that specialize in calculations on larger datasets that do not fit into memory and minimizing the memory required to run algorithms and using multicores.

This library implements Generalized Linear Models in both Julia and the D programming language. It attempts to create a comprehensive library that can handle larger datasets on multicore machines by dividing the computations to blocks that can be carried out in memory (for speed) and on disk (to conserve computational resources). It offers a variety of solvers that gives the user choice, flexibility and control, and also aims to be a fully comprehensive library in terms of post processing and to be comparable in performance with the best open source GLM solver libraries, comprehensive, convenient and simple to install and use on the Ubuntu (Linux) operating system.

## Prerequisites

* Julia, D, & R programming languages
* Openblas BLAS/LAPACK library

## Feature Development

Following is a list of items that need to be completed in order to finish the implementation of the GLM library intended
to be the replacement of bigReg.

Version 0.1 Core Functionality Implementation
---------------------------------------------------------------------------------------------------------------------
- [x] 1. Do a prototype of GLM and finish implementation of family and link functions, include weights and offsets. This
is a small scale implementation of the GLM. It is a prototype only where various optimizations of small scale regression
can be tried out. Link Functions:
  - [x] i.    Identity
  - [x] ii.   Inverse
  - [x] iii.  Log
  - [x] iv.   NegativeBinomial
  - [x] v.    Power
  - [x] vi.   Logit
  - [x] vii.  Probit
  - [x] viii. Cauchit
  - [x] ix.   OddsPower
  - [x] x.    LogCompliment
  - [x] xi.   LogLog
  - [x] xii.  ComplementaryLogLog

  Distributions:
  - [x] i.    Binomial
  - [x] ii.   Gamma
  - [x] iii.  Poisson
  - [x] iv.   Gaussian
  - [x] v.    InverseGaussian
  - [x] vi.   NegativeBinomial
  - [x] vii.  Power
  - [ ] viii. Tweedie
  - [ ] ix.   Ordinal Regression

  For Julia - [x] & D - [x]

- [x] 2. Do a speed optimization of the GLM prototype created in (1). Things to consider:
  - [x] i.  Matrix decomposition methods e.g. QR etc.
  - [ ] ii. ~~Optimise QR Speed by taking upper triangular R into account in the solve process.~~ Instead create various solver options using LAPACK linear equation solvers and least squares solvers. A very good website references is [Mark Gates](http://www.icl.utk.edu/~mgates3/) which has a good [routines list](http://www.icl.utk.edu/~mgates3/docs/lapack.html) documentation. It is well worth reading his lecture notes on dense linear algebra [part 1](http://www.icl.utk.edu/~mgates3/files/lect09-dla-2019.pdf) and [part 2](http://www.icl.utk.edu/~mgates3/files/lect10-dla-part2-2019.pdf). Also probably worth looking at least squares solve for LAPACK on [Netlib](https://www.netlib.org/lapack/lug/node27.html). The details follow, Linear Equation Solver Ax = b (square A):
    - [x] (a) `gesv` LU Decomposition Solver.
    - [x] (b) `posv` Cholesky Decomposition Solver.
    - [x] (c) `sysv` LDL Decomposition Solver.
    - [ ] (d) Could include `gesvxx`, `posvxx`, and `sysvxx` for more precise algorithms outputs and error bounds.
  There will be four options for least squares solvers min ||b - Av||_2:
    - [x] (a) `gels` Least squares solver using QR decomposition, requires *full rank* matrix A.
    - [x] (b) `gelsy` Orthogonal Factorization Solver.
    - [x] (c) `gelss` SVD Solver. (D Only)
    - [x] (d) `gelsd` SVD Solver divide & conquer.
  Matrix inverse algorithms (A^-1) to include:
    - [x] (a) `getri` LU Decomposition Inverse, `getrf` precursor.
    - [x] (b) `potri` Cholesky Decomposition Inverse, `potrf` precursor.
    - [x] (c) `sytri` LU Decomposition Inverse, `sytrf` precursor.
    - [x] (d) `svds` - My own name use SVD to do generalized inverse.
  Do this for 
    - [x] (a) D.
    - [x] (b) Julia.
  The Names used `GETRIInverse`, `POTRIInverse`, `SYTRFInverse`, `GESVDInverse`, `GESVSolver`, `POSVSolver`, `SYSVSolver`, `GELSSolver`, `GELSYSolver`, `GELSSSolver`, `GELSDSolver`.
  - [x] iii. Dispersion (phi) - **Done** function which you divide the `(XWX)^-1` matrix by to get the covariance matrix - **Done**. You will need to use page 110 of the Wood's GAM book, note that the Binomial and Poisson Distribution has `phi = 1`.
  - [ ] iv. Implement blocking matrix techniques using the techniques used in Golub's and Van Loan's Matrix Computations book.
    - [x] 1. 1-D block representation of matrices.
    - [ ] 2. 2-D block representation of matrices.
  - [ ] v. Implement parallel solver algorithms.
    - [X] 1. 1-D block representation of matrices.
    - [ ] 2. 2-D block representation of matrices.
  - [ ] vi. Include gradient descent solver.
  - [ ] vii. Look for further performance optimization, use **packed** format for symmetric matrix calls which would require fewer computations and could make a difference for problems with a large number of parameters.
  - [ ] viii. Compare your algorithm's performance with other implementations R, Python, Julia, H20, Scala Spark. If can show better or equal performance to all of these that would be a good start.
  - [ ] ix. Implement L-BFGS solver options.
  - [ ] x.  Implement data synthesis functions for GLM. Use Chapter 5 of Hardin & Hilbe and use this for benchmarking.
  - [ ] xi. Do you need a sparse solver? Investigate.
- [ ] 3. Implement memory and disk blocked matrix structure in Julia & D and
         integrate them with your current algorithm. Creating a generic interface that could contend with any data structure with the right methods returning the right types to the function(s).
- [ ] 4. Implement or adapt the current GLM algorithm to work with the memory and disk based blocked matrix data structures.
- [ ] 5. ~~Implement blocked data table for disk and in memory and their 
         mechanisms to be converted to your blocked data matrix structures. Use data.table, tibble, and r-frame as inspirations. There are many references to building data structures for instance you can search for lecture/notes on data table structures from reputable universities. Start by doing simple implementations as in your last package and then modify it with your new knowledge. There is no need to implement select, merge, sort algorithms. For now this blocked data table structure is simply for storing the data so that it can be converted to a blocked model matrix.~~ For now we can leverage R's `data.frames/data.table` structure and Julia's [DataFrames.jl package](http://juliadata.github.io/DataFrames.jl/v0.9/man/formulas/) which allows us to create model matrices. Therefore we can use binary IO of data table blocks to disk and memory using the current data table constructs in both languages. The main issue is whether we can do disk IO concurrently; in Julia concurrency is straightforward but concurrent disk IO is unknown. In R any computation or IO in R itself will be inefficient so later on we **must** create a native data table implementation in D which will hopefully allow us to fully parallelize the algorithm both on disk and in memory.
- [ ] 6. Write and finalise the documentation.

Version 0.2 Post-Processing & Model Search Implementation
---------------------------------------------------------------------------------------------------------------------

- [ ] 1. Create summary function complete with pretty printing for the model output.
- [ ] 2. Create diagnostic plotting functions for the model outputs.
- [ ] 3. Measures/Tests such as X2, Significance tests, AIC/BIC, R^2, and so on.
- [ ] 4. Model comparisons, T-tests, ANOVA and so forth. Refer to the model comparisons package in 
         R for inspiration.
- [ ] 5. Write an *update()* function for modifying the model and write a *step()* function for model
      searches, forward, backward, both directional searches.
- [ ] 6. Write/finalize the documentation.
- [ ] 7. Do you need to worry about which operating system this will be run on? Theoretically, if it is written
         in Julia, D and R it could be that you won't need to worry about this.

Version 1.0 Alpha
----------------------------------------------------------------------------------------------------------------------
- [ ] 1. Make sure that all the functionality is working as designed and are all properly tested and documented.

Version 1.0 Beta
----------------------------------------------------------------------------------------------------------------------
- [ ] 1. Release for testing and carry out bug fixing - Github ony

Version 1.0 Release Candidate
----------------------------------------------------------------------------------------------------------------------
- [ ] 1. Write a presentation about this package and start publicising in Active Analytics website and do presentations
   about it. Do any further changes that need to be done.
- [ ] 2. Attempt to release this on CRAN

Version 1.0
----------------------------------------------------------------------------------------------------------------------
- [ ] 1. By now everything should be baked in and should be stable. Release it and enjoy using it.

Version 1.1
----------------------------------------------------------------------------------------------------------------------
- [ ] 1. Add Constraints to regression, use Generalized QR decomposition.
- [ ] 2. Add L1, and L-Infinity error functions for regression.
- [ ] 3. Include regression constraints both for linear regression and GLM using LAPACK routines for [generalized least squares (MKL)](https://software.intel.com/en-us/mkl-developer-reference-fortran-generalized-linear-least-squares-lls-problems-lapack-driver-routines), see [Netlib](https://www.netlib.org/lapack/lug/node28.html) also.
- [ ] 4. Add regularization feature.
- [ ] 5. Multiple Y variables? LAPACK allows this feature to be implemented to the framework.
