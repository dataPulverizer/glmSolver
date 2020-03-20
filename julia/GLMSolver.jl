module GLMSolver

path = "/home/chib/code/glmSolver/"
include(path * "julia/io.jl")
include(path * "julia/link.jl")
include(path * "julia/family.jl")
include(path * "julia/solver.jl")
include(path * "julia/tools.jl")
include(path * "julia/fit.jl")
include(path * "julia/sample.jl")
include(path * "julia/simulate.jl")

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
       AbstractGradientDescentSolver,
       GradientDescentSolver,
       MomentumSolver,
       NesterovSolver,
       AdagradSolver,
       AdadeltaSolver,
       RMSpropSolver,
       AdamSolver,
       AdaMaxSolver,
       NAdamSolver,
       AMSGradSolver,
       XWX,
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
       glm,
       generateRandomMatrix,
       cov2cor,
       cor2cov,
       mvrnorm,
       AbstractDistribution,
       BetaDistribution,
       sample,
       UniformDistribution,
       min,
       max,
       range,
       I,
       AbstractRandomCorrelationMatrix,
       BetaGenerator,
       OnionGenerator,
       UniformGenerator,
       VineGenerator,
       randomCorrelationMatrix,
       simulateData
       AbstractPoissonDistribution
       PoissonDistribution

end # module
