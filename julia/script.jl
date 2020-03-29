#=
  Demos for various glm functions
=#
path = "/home/chib/code/glmSolver/"

include(path * "julia/GLMSolver.jl")
using .GLMSolver

# Load the data set
# include(path * "julia/data.jl");

using Random: seed!

seed!(0);
gammaX, gammaY = simulateData(Float64, GammaDistribution(), LogLink(), 30, 10_000);
gammaBlockX = matrixToBlock(gammaX, 10);
gammaBlockY = matrixToBlock(gammaY, 10);

binomialX, binomialY = simulateData(Float64, BinomialDistribution(), LogLink(), 30, 10_000);
binomialBlockX = matrixToBlock(binomialX, 10);
binomialBlockY = matrixToBlock(binomialY, 10);

poissonX, poissonY = simulateData(Float64, BinomialDistribution(), LogLink(), 30, 10_000);
poissonBlockX = matrixToBlock(poissonX, 10);
poissonBlockY = matrixToBlock(poissonY, 10);

# Fit the models
gamma_distrib_log_link = glm(RegularData(), gammaX, gammaY, GammaDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse());
gamma_distrib_inverse_link = glm(RegularData(), gammaX, gammaY, GammaDistribution(), InverseLink(), GESVSolver(), inverse = GETRIInverse());
gamma_distrib_identity_link = glm(RegularData(), gammaX, gammaY, GammaDistribution(), IdentityLink(), GESVSolver(),  inverse = GETRIInverse());

gamma_distrib_power_model_1 = glm(RegularData(), gammaX, gammaY, GammaDistribution(), PowerLink(0.0), GESVSolver(), inverse = GETRIInverse());
gamma_distrib_power_model_2 = glm(RegularData(), gammaX, gammaY, GammaDistribution(), PowerLink(1/3), GESVSolver(), inverse = GETRIInverse());
gamma_distrib_negative_binomial_link_1 = glm(RegularData(), gammaX, gammaY, GammaDistribution(), NegativeBinomialLink(1.0), GESVSolver(), inverse = GETRIInverse());
gamma_distrib_negative_binomial_link_2 = glm(RegularData(), gammaX, gammaY, GammaDistribution(), NegativeBinomialLink(2.0), GESVSolver(), inverse = GETRIInverse());

#= Binomial Distribution With Logit Link =#
binomial_logit_link = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), LogitLink(), GESVSolver(), inverse = GETRIInverse());
#= Binomial Distribution With Probit Link =#
binomial_probit_link = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), ProbitLink(), GESVSolver(), inverse = GETRIInverse());
binomial_cauchit_link = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), CauchitLink(), GESVSolver(), inverse = GETRIInverse());
binomial_oddspower_link_1 = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), inverse = GETRIInverse());

binomial_distrib_odds_power_link_1 = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), inverse = GETRIInverse());
# binomial_distrib_odds_power_link_2 = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), inverse = GETRIInverse())

# binomial_oddspower_link_2 = glm(RegularData(), binomialX, binomialY, BinomialDistribution, OddsPowerLink{1})

# bernoulli_logcomplementary = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), LogComplimentLink(), GESVSolver(), inverse = GETRIInverse())
bernoulli_loglog = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), LogLogLink(), GESVSolver(), inverse = GETRIInverse());
# bernoulli_complementaryloglog = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), ComplementaryLogLogLink(), GESVSolver(), inverse = GETRIInverse())

# LogLink With Gaussian Distribution
log_link_gaussian_distrib = glm(RegularData(), gammaX, gammaY, GaussianDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())

# LogLink With Gamma Distribution
log_link_gamma_distribution = glm(RegularData(), gammaX, gammaY, GammaDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
log_link_inversegaussian_distribution = glm(RegularData(), gammaX, gammaY, InverseGaussianDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())

log_link_poisson_distribution = glm(RegularData(), poissonX, poissonY, PoissonDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
logit_link_bernoulli_distrib = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), LogitLink(), GESVSolver(), inverse = GETRIInverse())

log_link_negative_bernoulli_distrib = glm(RegularData(), gammaX, gammaY, NegativeBernoulliDistribution(0.5), LogLink(), GESVSolver(), inverse = GETRIInverse())
cauchit_link_binomial_distribution = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), CauchitLink(), GESVSolver(), inverse = GETRIInverse())

#=======================================================================================#
igaussianLogModel = glm(RegularData(), gammaX, gammaY, InverseGaussianDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
igaussianInverseModel = glm(RegularData(), gammaX, gammaY, InverseGaussianDistribution(), InverseLink(), GESVSolver(), inverse = GETRIInverse()) 

bernoulliLogit = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), LogitLink(), GESVSolver(), inverse = GETRIInverse())
bernoulliProbit = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), ProbitLink(), GESVSolver(), inverse = GETRIInverse())

bernoulliOddsPower0 = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), inverse = GETRIInverse())

# bernoulliLogComplementary = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), LogComplimentLink(), GESVSolver(), inverse = GETRIInverse())
bernoulliLogLog = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), LogLogLink(), GESVSolver(), inverse = GETRIInverse())

# 
# bernoulliComplementaryLogLog = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), ComplementaryLogLogLink(), GESVSolver(), inverse = GETRIInverse())

# Count Data
poissonLog = glm(RegularData(), poissonX, poissonY, PoissonDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
# Here the parameter is the inverse of what it is in R
# negativeBernoulliLog = glm(RegularData(), poissonX, poissonY, NegativeBernoulliDistribution(1/2), LogLink(), GESVSolver(), inverse = GETRIInverse())

# Education for full binomial data
binomialLogitModel = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), LogitLink(), GESVSolver(), inverse = GETRIInverse())

# Testing the Cauchit Link function
binomialCauchit = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), CauchitLink(), GESVSolver(), inverse = GETRIInverse())

#=
TODO:

1. Do test/examples for offset and weights
=#

# Block Demos
block_gamma_distrib_log_link = glm(Block1D(), gammaBlockX, gammaBlockY, GammaDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
block_gamma_distrib_inverse_link = glm(Block1D(), gammaBlockX, gammaBlockY, GammaDistribution(), InverseLink(), GESVSolver(), inverse = GETRIInverse()) 
block_gamma_distrib_identity_link = glm(Block1D(), gammaBlockX, gammaBlockY, GammaDistribution(), IdentityLink(), GESVSolver(), inverse = GETRIInverse())
block_gamma_distrib_power_model_1 = glm(Block1D(), gammaBlockX, gammaBlockY, GammaDistribution(), PowerLink(0.0), GESVSolver(), inverse = GETRIInverse())
block_gamma_distrib_negative_binomial_link_2 = glm(Block1D(), gammaBlockX, gammaBlockY, GammaDistribution(), NegativeBinomialLink(2.0), GESVSolver(), inverse = GETRIInverse())
block_log_link_gaussian_distrib = glm(Block1D(), gammaBlockX, gammaBlockY, GaussianDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
block_igaussianLogModel = glm(Block1D(), gammaBlockX, gammaBlockY, InverseGaussianDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
block_igaussianInverseModel = glm(Block1D(), gammaBlockX, gammaBlockY, InverseGaussianDistribution(), InverseLink(), GESVSolver(), inverse = GETRIInverse()) 
# block_binomial_oddspower_link_1 = glm(Block1D(), binomialBlockX, binomialBlockY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), inverse = GETRIInverse())
# block_binomial_distrib_odds_power_link_2 = glm(Block1D(), binomialBlockX, binomialBlockY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), inverse = GETRIInverse())
block_cauchit_link_binomial_distribution = glm(Block1D(), binomialBlockX, binomialBlockY, BinomialDistribution(), CauchitLink(), GESVSolver(), inverse = GETRIInverse())
block_poissonLog = glm(Block1D(), poissonBlockX, poissonBlockY, PoissonDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())

block_logit_link_bernoulli_distrib = glm(Block1D(), binomialBlockX, binomialBlockY, BinomialDistribution(), LogitLink(), GESVSolver(), inverse = GETRIInverse())
block_binomial_distrib_odds_power_link_1 = glm(Block1D(), binomialBlockX, binomialBlockY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), inverse = GETRIInverse())
block_bernoulliLogit = glm(Block1D(), binomialBlockX, binomialBlockY, BinomialDistribution(), LogitLink(), GESVSolver(), inverse = GETRIInverse())
block_bernoulliOddsPower0 = glm(Block1D(), binomialBlockX, binomialBlockY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), inverse = GETRIInverse())


block_gamma_distrib_log_link.coefficients |> println
block_gamma_distrib_inverse_link.coefficients |> println
block_gamma_distrib_identity_link.coefficients |> println
block_gamma_distrib_power_model_1.coefficients |> println
block_gamma_distrib_negative_binomial_link_2.coefficients |> println
block_log_link_gaussian_distrib.coefficients |> println
block_igaussianLogModel.coefficients |> println
block_igaussianInverseModel.coefficients |> println
block_cauchit_link_binomial_distribution.coefficients |> println
block_poissonLog.coefficients |> println

block_logit_link_bernoulli_distrib.coefficients |> println
block_binomial_distrib_odds_power_link_1.coefficients |> println
block_bernoulliLogit.coefficients |> println
block_bernoulliOddsPower0.coefficients |> println


gamma_distrib_log_link = glm(RegularData(), gammaX, gammaY, GammaDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
gamma_distrib_inverse_link = glm(RegularData(), gammaX, gammaY, GammaDistribution(), InverseLink(), GESVSolver(), inverse = GETRIInverse()) 
gamma_distrib_identity_link = glm(RegularData(), gammaX, gammaY, GammaDistribution(), IdentityLink(), GESVSolver(), inverse = GETRIInverse())
gamma_distrib_power_model_1 = glm(RegularData(), gammaX, gammaY, GammaDistribution(), PowerLink(0.0), GESVSolver(), inverse = GETRIInverse())
gamma_distrib_negative_binomial_link_2 = glm(RegularData(), gammaX, gammaY, GammaDistribution(), NegativeBinomialLink(2.0), GESVSolver(), inverse = GETRIInverse())
log_link_gaussian_distrib = glm(RegularData(), gammaX, gammaY, GaussianDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
igaussianLogModel = glm(RegularData(), gammaX, gammaY, InverseGaussianDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
igaussianInverseModel = glm(RegularData(), gammaX, gammaY, InverseGaussianDistribution(), InverseLink(), GESVSolver(), inverse = GETRIInverse()) 
# binomial_oddspower_link_1 = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), inverse = GETRIInverse())
# binomial_distrib_odds_power_link_2 = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), inverse = GETRIInverse())
cauchit_link_binomial_distribution = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), CauchitLink(), GESVSolver(), inverse = GETRIInverse())
poissonLog = glm(RegularData(), poissonX, poissonY, PoissonDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())

logit_link_bernoulli_distrib = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), LogitLink(), GESVSolver(), inverse = GETRIInverse())
binomial_distrib_odds_power_link_1 = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), inverse = GETRIInverse())
bernoulliLogit = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), LogitLink(), GESVSolver(), inverse = GETRIInverse())
bernoulliOddsPower0 = glm(RegularData(), binomialX, binomialY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), inverse = GETRIInverse())


gamma_distrib_log_link.coefficients |> println
gamma_distrib_inverse_link.coefficients |> println
gamma_distrib_identity_link.coefficients |> println
gamma_distrib_power_model_1.coefficients |> println
gamma_distrib_negative_binomial_link_2.coefficients |> println
log_link_gaussian_distrib.coefficients |> println
igaussianLogModel.coefficients |> println
igaussianInverseModel.coefficients |> println
binomial_oddspower_link_1.coefficients |> println
cauchit_link_binomial_distribution.coefficients |> println
poissonLog.coefficients |> println

logit_link_bernoulli_distrib.coefficients |> println
binomial_distrib_odds_power_link_1.coefficients |> println
bernoulliLogit.coefficients |> println
bernoulliOddsPower0.coefficients |> println

# Parallel Block Algorithm
block_gamma_distrib_log_link = glm(Block1D(), gammaBlockX, gammaBlockY, GammaDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())
block_parallel_gamma_distrib_log_link = glm(Block1DParallel(), gammaBlockX, gammaBlockY, GammaDistribution(), LogLink(), GESVSolver(), inverse = GETRIInverse())

# block_binomial_oddspower_link_1 = glm(Block1D(), binomialBlockX, binomialBlockY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), inverse = GETRIInverse())
# block_parallel_binomial_oddspower_link_1 = glm(Block1DParallel(), binomialBlockX, binomialBlockY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), inverse = GETRIInverse())

block_logit_link_bernoulli_distrib = glm(Block1D(), binomialBlockX, binomialBlockY, BinomialDistribution(), LogitLink(), GESVSolver(), inverse = GETRIInverse())
block_parallel_logit_link_bernoulli_distrib = glm(Block1DParallel(), binomialBlockX, binomialBlockY, BinomialDistribution(), LogitLink(), GESVSolver(), inverse = GETRIInverse())

