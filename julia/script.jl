#=
  Demos for various glm functions
=#
path = "/home/chib/code/glmSolver/"

include(path * "julia/GLMSolver.jl")
using .GLMSolver

# Load the data set
include(path * "julia/data.jl");

# Fit the models
# TODO: Fix the things that are not converging
gamma_distrib_log_link = glm(RegularData(), energyX, energyY, GammaDistribution(), LogLink(), GESVSolver(), GETRIInverse())
gamma_distrib_inverse_link = glm(RegularData(), energyX, energyY, GammaDistribution(), InverseLink(), GESVSolver(), GETRIInverse()) 
gamma_distrib_identity_link = glm(RegularData(), energyX, energyY, GammaDistribution(), IdentityLink(), GESVSolver(), GETRIInverse())

gamma_distrib_power_model_1 = glm(RegularData(), energyX, energyY, GammaDistribution(), PowerLink(0.0), GESVSolver(), GETRIInverse())
gamma_distrib_power_model_2 = glm(RegularData(), carsX, carsY, GammaDistribution(), PowerLink(1/3), GESVSolver(), GETRIInverse())
gamma_distrib_negative_binomial_link_1 = glm(RegularData(), carsX, carsY, GammaDistribution(), NegativeBinomialLink(1.0), GESVSolver(), GETRIInverse())
gamma_distrib_negative_binomial_link_2 = glm(RegularData(), energyX, energyY, GammaDistribution(), NegativeBinomialLink(2.0), GESVSolver(), GETRIInverse())

#= Binomial Distribution With Logit Link =#
binomial_logit_link = glm(RegularData(), creditX, creditY, BinomialDistribution(), LogitLink(), GESVSolver(), GETRIInverse())
#= Binomial Distribution With Probit Link =#
binomial_probit_link = glm(RegularData(), gpaX, gpaY, BinomialDistribution(), ProbitLink(), GESVSolver(), GETRIInverse())
binomial_cauchit_link = glm(RegularData(), gpaX, gpaY, BinomialDistribution(), CauchitLink(), GESVSolver(), GETRIInverse())
binomial_oddspower_link_1 = glm(RegularData(), educationX, educationY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), GETRIInverse())

binomial_distrib_odds_power_link_1 = glm(RegularData(), creditX, creditY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), GETRIInverse())
binomial_distrib_odds_power_link_2 = glm(RegularData(), educationX, educationY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), GETRIInverse())

# binomial_oddspower_link_2 = glm(RegularData(), creditX, creditY, BinomialDistribution, OddsPowerLink{1})
bernoulli_logcomplementary = glm(RegularData(), gpaX, gpaY, BinomialDistribution(), LogComplimentLink(), GESVSolver(), GETRIInverse())
bernoulli_loglog = glm(RegularData(), gpaX, gpaY, BinomialDistribution(), LogLogLink(), GESVSolver(), GETRIInverse())
bernoulli_complementaryloglog = glm(RegularData(), gpaX, gpaY, BinomialDistribution(), ComplementaryLogLogLink(), GESVSolver(), GETRIInverse())

# LogLink With Gaussian Distribution
log_link_gaussian_distrib = glm(RegularData(), energyX, energyY, GaussianDistribution(), LogLink(), GESVSolver(), GETRIInverse())

# LogLink With Gamma Distribution
log_link_gamma_distribution = glm(RegularData(), energyX, energyY, GammaDistribution(), LogLink(), GESVSolver(), GETRIInverse())
log_link_inversegaussian_distribution = glm(RegularData(), energyX, energyY, InverseGaussianDistribution(), LogLink(), GESVSolver(), GETRIInverse())
log_link_poisson_distribution = glm(RegularData(), energyX, energyY, PoissonDistribution(), LogLink(), GESVSolver(), GETRIInverse())
logit_link_bernoulli_distrib = glm(RegularData(), creditX, creditY, BinomialDistribution(), LogitLink(), GESVSolver(), GETRIInverse())
log_link_negative_bernoulli_distrib = glm(RegularData(), energyX, energyY, NegativeBernoulliDistribution(0.5), LogLink(), GESVSolver(), GETRIInverse())
log_link_power_distrib = glm(RegularData(), carsX, carsY, PowerDistribution(0.5), PowerLink(0.5), GESVSolver(), GETRIInverse())
logit_link_binomial_distribution = glm(RegularData(), educationX, educationY, BinomialDistribution(), LogLink(), GESVSolver(), GETRIInverse())
cauchit_link_binomial_distribution = glm(RegularData(), educationX, educationY, BinomialDistribution(), CauchitLink(), GESVSolver(), GETRIInverse())

#=======================================================================================#
igaussianLogModel = glm(RegularData(), energyX, energyY, InverseGaussianDistribution(), LogLink(), GESVSolver(), GETRIInverse())
igaussianInverseModel = glm(RegularData(), energyX, energyY, InverseGaussianDistribution(), InverseLink(), GESVSolver(), GETRIInverse()) 

bernoulliLogit = glm(RegularData(), creditX, creditY, BinomialDistribution(), LogitLink(), GESVSolver(), GETRIInverse())
bernoulliProbit = glm(RegularData(), gpaX, gpaY, BinomialDistribution(), ProbitLink(), GESVSolver(), GETRIInverse())

bernoulliOddsPower0 = glm(RegularData(), creditX, creditY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), GETRIInverse())

# Need to get examples for this to make this link function behave better
bernoulliOddsPower1 = glm(RegularData(), educationX, educationY, BinomialDistribution(), OddsPowerLink(1.0), GESVSolver(), GETRIInverse())


bernoulliLogComplementary = glm(RegularData(), gpaX, gpaY, BinomialDistribution(), LogComplimentLink(), GESVSolver(), GETRIInverse())
bernoulliLogLog = glm(RegularData(), gpaX, gpaY, BinomialDistribution(), LogLogLink(), GESVSolver(), GETRIInverse())
bernoulliComplementaryLogLog = glm(RegularData(), gpaX, gpaY, BinomialDistribution(), ComplementaryLogLogLink(), GESVSolver(), GETRIInverse())

# Count Data
poissonLog = glm(RegularData(), insuranceX, insuranceY, PoissonDistribution(), LogLink(), GESVSolver(), GETRIInverse())
# Here the parameter is the inverse of what it is in R
negativeBernoulliLog = glm(RegularData(), quineX, quineY, NegativeBernoulliDistribution(1/2), LogLink(), GESVSolver(), GETRIInverse())

# Education for full binomial data
binomialLogitModel = glm(RegularData(), educationX, educationY, BinomialDistribution(), LogitLink(), GESVSolver(), GETRIInverse())

# Testing the Cauchit Link function
binomialCauchit = glm(RegularData(), educationX, educationY, BinomialDistribution(), CauchitLink(), GESVSolver(), GETRIInverse())

#=
TODO:

1. Do test/examples for offset and weights
=#

# Block Demos
block_gamma_distrib_log_link = glm(Block1D(), energyBlockX, energyBlockY, GammaDistribution(), LogLink(), GESVSolver(), GETRIInverse())
block_gamma_distrib_inverse_link = glm(Block1D(), energyBlockX, energyBlockY, GammaDistribution(), InverseLink(), GESVSolver(), GETRIInverse()) 
block_gamma_distrib_identity_link = glm(Block1D(), energyBlockX, energyBlockY, GammaDistribution(), IdentityLink(), GESVSolver(), GETRIInverse())
block_gamma_distrib_power_model_1 = glm(Block1D(), energyBlockX, energyBlockY, GammaDistribution(), PowerLink(0.0), GESVSolver(), GETRIInverse())
block_gamma_distrib_negative_binomial_link_2 = glm(Block1D(), energyBlockX, energyBlockY, GammaDistribution(), NegativeBinomialLink(2.0), GESVSolver(), GETRIInverse())
block_log_link_gaussian_distrib = glm(Block1D(), energyBlockX, energyBlockY, GaussianDistribution(), LogLink(), GESVSolver(), GETRIInverse())
block_igaussianLogModel = glm(Block1D(), energyBlockX, energyBlockY, InverseGaussianDistribution(), LogLink(), GESVSolver(), GETRIInverse())
block_igaussianInverseModel = glm(Block1D(), energyBlockX, energyBlockY, InverseGaussianDistribution(), InverseLink(), GESVSolver(), GETRIInverse()) 
block_binomial_oddspower_link_1 = glm(Block1D(), educationBlockX, educationBlockY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), GETRIInverse())
block_binomial_distrib_odds_power_link_2 = glm(Block1D(), educationBlockX, educationBlockY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), GETRIInverse())
block_logit_link_binomial_distribution = glm(Block1D(), educationBlockX, educationBlockY, BinomialDistribution(), LogLink(), GESVSolver(), GETRIInverse())
block_cauchit_link_binomial_distribution = glm(Block1D(), educationBlockX, educationBlockY, BinomialDistribution(), CauchitLink(), GESVSolver(), GETRIInverse())
block_poissonLog = glm(Block1D(), insuranceBlockX, insuranceBlockY, PoissonDistribution(), LogLink(), GESVSolver(), GETRIInverse())

block_logit_link_bernoulli_distrib = glm(Block1D(), creditBlockX, creditBlockY, BinomialDistribution(), LogitLink(), GESVSolver(), GETRIInverse())
block_binomial_distrib_odds_power_link_1 = glm(Block1D(), creditBlockX, creditBlockY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), GETRIInverse())
block_bernoulliLogit = glm(Block1D(), creditBlockX, creditBlockY, BinomialDistribution(), LogitLink(), GESVSolver(), GETRIInverse())
block_bernoulliOddsPower0 = glm(Block1D(), creditBlockX, creditBlockY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), GETRIInverse())


block_gamma_distrib_log_link.coefficients |> println
block_gamma_distrib_inverse_link.coefficients |> println
block_gamma_distrib_identity_link.coefficients |> println
block_gamma_distrib_power_model_1.coefficients |> println
block_gamma_distrib_negative_binomial_link_2.coefficients |> println
block_log_link_gaussian_distrib.coefficients |> println
block_igaussianLogModel.coefficients |> println
block_igaussianInverseModel.coefficients |> println
block_binomial_oddspower_link_1.coefficients |> println
block_binomial_distrib_odds_power_link_2.coefficients |> println
block_logit_link_binomial_distribution.coefficients |> println
block_cauchit_link_binomial_distribution.coefficients |> println
block_poissonLog.coefficients |> println

block_logit_link_bernoulli_distrib.coefficients |> println
block_binomial_distrib_odds_power_link_1.coefficients |> println
block_bernoulliLogit.coefficients |> println
block_bernoulliOddsPower0.coefficients |> println


gamma_distrib_log_link = glm(RegularData(), energyX, energyY, GammaDistribution(), LogLink(), GESVSolver(), GETRIInverse())
gamma_distrib_inverse_link = glm(RegularData(), energyX, energyY, GammaDistribution(), InverseLink(), GESVSolver(), GETRIInverse()) 
gamma_distrib_identity_link = glm(RegularData(), energyX, energyY, GammaDistribution(), IdentityLink(), GESVSolver(), GETRIInverse())
gamma_distrib_power_model_1 = glm(RegularData(), energyX, energyY, GammaDistribution(), PowerLink(0.0), GESVSolver(), GETRIInverse())
gamma_distrib_negative_binomial_link_2 = glm(RegularData(), energyX, energyY, GammaDistribution(), NegativeBinomialLink(2.0), GESVSolver(), GETRIInverse())
log_link_gaussian_distrib = glm(RegularData(), energyX, energyY, GaussianDistribution(), LogLink(), GESVSolver(), GETRIInverse())
igaussianLogModel = glm(RegularData(), energyX, energyY, InverseGaussianDistribution(), LogLink(), GESVSolver(), GETRIInverse())
igaussianInverseModel = glm(RegularData(), energyX, energyY, InverseGaussianDistribution(), InverseLink(), GESVSolver(), GETRIInverse()) 
binomial_oddspower_link_1 = glm(RegularData(), educationX, educationY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), GETRIInverse())
binomial_distrib_odds_power_link_2 = glm(RegularData(), educationX, educationY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), GETRIInverse())
logit_link_binomial_distribution = glm(RegularData(), educationX, educationY, BinomialDistribution(), LogLink(), GESVSolver(), GETRIInverse())
cauchit_link_binomial_distribution = glm(RegularData(), educationX, educationY, BinomialDistribution(), CauchitLink(), GESVSolver(), GETRIInverse())
poissonLog = glm(RegularData(), insuranceX, insuranceY, PoissonDistribution(), LogLink(), GESVSolver(), GETRIInverse())

logit_link_bernoulli_distrib = glm(RegularData(), creditX, creditY, BinomialDistribution(), LogitLink(), GESVSolver(), GETRIInverse())
binomial_distrib_odds_power_link_1 = glm(RegularData(), creditX, creditY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), GETRIInverse())
bernoulliLogit = glm(RegularData(), creditX, creditY, BinomialDistribution(), LogitLink(), GESVSolver(), GETRIInverse())
bernoulliOddsPower0 = glm(RegularData(), creditX, creditY, BinomialDistribution(), OddsPowerLink(0.0), GESVSolver(), GETRIInverse())


gamma_distrib_log_link.coefficients |> println
gamma_distrib_inverse_link.coefficients |> println
gamma_distrib_identity_link.coefficients |> println
gamma_distrib_power_model_1.coefficients |> println
gamma_distrib_negative_binomial_link_2.coefficients |> println
log_link_gaussian_distrib.coefficients |> println
igaussianLogModel.coefficients |> println
igaussianInverseModel.coefficients |> println
binomial_oddspower_link_1.coefficients |> println
binomial_distrib_odds_power_link_2.coefficients |> println
logit_link_binomial_distribution.coefficients |> println
cauchit_link_binomial_distribution.coefficients |> println
poissonLog.coefficients |> println

logit_link_bernoulli_distrib.coefficients |> println
binomial_distrib_odds_power_link_1.coefficients |> println
bernoulliLogit.coefficients |> println
bernoulliOddsPower0.coefficients |> println

# Parallel Block Algorithm
block_gamma_distrib_log_link = glm(Block1D(), energyBlockX, energyBlockY, GammaDistribution(), LogLink(), GESVSolver(), GETRIInverse())
block_parallel_gamma_distrib_log_link = glm(Block1DParallel(), energyBlockX, energyBlockY, GammaDistribution(), LogLink(), GESVSolver(), GETRIInverse())


block_binomial_oddspower_link_1 = glm(Block1D(), educationBlockX, educationBlockY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), GETRIInverse())
block_parallel_binomial_oddspower_link_1 = glm(Block1DParallel(), educationBlockX, educationBlockY, BinomialDistribution(), OddsPowerLink(2.0), GESVSolver(), GETRIInverse())


block_logit_link_bernoulli_distrib = glm(Block1D(), creditBlockX, creditBlockY, BinomialDistribution(), LogitLink(), GESVSolver(), GETRIInverse())
block_parallel_logit_link_bernoulli_distrib = glm(Block1DParallel(), creditBlockX, creditBlockY, BinomialDistribution(), LogitLink(), GESVSolver(), GETRIInverse())




