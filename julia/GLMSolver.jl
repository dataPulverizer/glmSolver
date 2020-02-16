module GLMSolver

path = "/home/chib/code/glmSolver/"
include(path * "julia/io.jl")
include(path * "julia/link.jl")
include(path * "julia/family.jl")
include(path * "julia/solver.jl")
include(path * "julia/tools.jl")
include(path * "julia/fit.jl")

export write2DArray,
       writeBlockMatrix,
       read2DArray,
       readBlockMatrix,
       writeNDArray,
       readNDArray,
       AbstractLink,
       IdentityLink,
       LogLink,
       InverseLink,
       NegativeBinomialLink,
       LogitLink,
       ProbitLink,
       CauchitLink,
       OddsPowerLink,
       LogComplimentLink,
       LogLogLink,
       ComplementaryLogLogLink,
       PowerLink,
       linkfun,
       deta_dmu,
       linkinv,
       AbstractDistribution,
       BernoulliDistribution,
       BinomialDistribution,
       GammaDistribution,
       PoissonDistribution,
       GaussianDistribution,
       InverseGaussianDistribution,
       NegativeBernoulliDistribution,
       PowerDistribution,
       init!,
       variance,
       devianceResiduals,
       AbstractSolver,
       VanillaSolver,
       QRSolver,
       solve,
       AbstractInverse,
       GETRIInverse,
       POTRIInverse,
       SYTRFInverse,
       GESVDInverse,
       inv,
       GESVSolver,
       POSVSolver,
       SYSVSolver,
       GELSSolver,
       GELSYSolver,
       GELSDSolver,
       solve!,
       cov,
       Z,
       W,
       Control,
       absoluteError,
       relativeError,
       AbstractMatrixType,
       RegularData,
       AbstractGLM,
       GLM,
       GLMBlock1D,
       Block1D,
       Block1DParallel,
       glm
end # module
