module GLMSolver

path = "/home/chib/code/glmSolver/"
include(path * "julia/link.jl")
include(path * "julia/family.jl")
include(path * "julia/tools.jl")
include(path * "julia/fit.jl")


export AbstractLink,
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
       Z,
       W,
       Control,
       absoluteError,
       relativeError,
       GLM,
       glm

end # module