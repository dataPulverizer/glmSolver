# GLMSolver
Solvers for GLM that specialize in calculations on large datasets

## Feature Development

Following is a list of items that need to be completed in order to finish the implementation of the GLM library intended
to be the replacement of bigReg.

Version 0.1 Core Functionality Implementation
---------------------------------------------------------------------------------------------------------------------
- [x] 1. Do a prototype of GLM and finish implementation of family and link functions, include weights and offsets. This
is a small scale implementation of the GLM. It is a prototype only where various optimizations of small scale regression
can be tried out.
- [ ] 2. Do a speed optimization of the GLM prototype created in (1). Things to consider:
  - [x] i.  Matrix decomposition methods e.g. QR etc.
  - [ ] ii. Optimise QR Speed by taking upper triangular R into account in the solve process.
  - [ ] iii. Create X2 and Dispersion (phi) function which you divide the `(XWX)^-1` matrix by to get the
             covariance matrix. You will need to use page 110 of the Wood's GAM book, note that the Binomial
             and Poisson Distribution has `phi = 1`.
  - [ ] iv. Look at Julia's GLM implementation, they use matrix multiplications using inplace modification
            of arrays which could be faster? e.g. *mul!*
  - [ ] v. Implement blocking matrix techniques using the techniques used in Golub's and Van Loan's 
            Matrix Computations book.
  - [ ] vi. Compare your algorithm's performance with other implementations R, Python, Julia, H20, Scala Spark. If
            can show better or equal performance to all of these that would be a good start.
  - [ ] vii. Implement L-BFGS and gradient descent as options in addition to the standard Fisher Matrix/Hessian solver.
  - [ ] viii.  Implement data synthesis functions for GLM. Use Chapter 5 of Hardin & Hilbe and use this for
            benchmarking.
  - [ ] ix. Do you need a sparse solver? Investigate.
- [ ] 3. Implement memory and disk blocked matrix structure in Julia (and Rust if applicable) and integrate them 
         with your current algorithm. Creating a generic interface that could contend with any data structure with
         the right methods returning the right types to the function(s).
- [ ] 4. Implement or adapt the current GLM algorithm to work with the memory and disk based blocked matrix data 
         structures.
- [ ] 5. Implement blocked data table for disk and in memory and their mechanisms to be converted to your blocked
         data matrix structures. Use data.table, tibble, and r-frame as inspirations. There are many references
         to building data structures for instance you can search for lecture/notes on data table structures from
         reputable universities. Start by doing simple implementations as in your last package and then modify
         it with your new knowledge. There is no need to implement select, merge, sort algorithms. For now this
         blocked data table structure is simply for storing the data so that it can be converted to a blocked
         model matrix.
- [ ] 6. Write and finalise the documentation.

Version 0.2 Post-Processing & Model Search Implementation
---------------------------------------------------------------------------------------------------------------------

- [ ] 1. Create summary function complete with pretty printing for the model output.
- [ ] 2. Create diagnostic plotting functions for the model outputs.
- [ ] 3. Measures/Tests such as AIC/BIC, R^2, and so on.
- [ ] 4. Model comparisons, T-tests, ANOVA and so forth. Refer to the model comparisons package in 
         R for inspiration.
- [ ] 5. Write an *update()* function for modifying the model and write a *step()* function for model
      searches, forward, backward, both directional searches.
- [ ] 6. Write/finalize the documentation.
- [ ] 7. Do you need to worry about which operating system this will be run on? Theoretically, if it is written
         in Julia, Rust and R it could be that you won't need to worry about this.

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
- [ ] 2. Add L1, and L Infinity error functions for regression.
