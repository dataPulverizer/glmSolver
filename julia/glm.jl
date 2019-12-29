#=
  Basic glm implementation in Julia
=#

using Distributions

# Defining the standard normal for further use
const standardNormal = Normal()

# Standard Normal pdf
function pdfSN(eta::T)::T where {T <: AbstractFloat}
  return exp(-(eta^2)/2)/sqrt(2*pi)
end


# TODO:
# 1. Rearrange the Link and Family Functions so that all the 
#    functions for each link or family functions are grouped
#    together rather than groups of the same functions. Done
# 2. Replace the abstract classes for actual family and link
#    functions with concrete classes and remove parameters
#    that are part of the class type and put them in the
#    struct. Done
# 3. I think it is time to convert Float64 to a type 
#    parameter. Done
# 4. Use x^-1 rather than 1/x
# 5. Precompile the function: precompile(f, args::Tuple{Vararg{Any}})
# 6. Abstract away Distributions package implement your own standard
#    normal. erfc etc.
#

# Compliment of epsilon 1 - eps(Float)
function ceps(T::Type{<: AbstractFloat})
  return 1 - eps(T)
end

#= Abstract GLM Types =#
abstract type AbstractGLM end

#=
  Types for Link Functions
=#
abstract type AbstractLink end
struct IdentityLink <: AbstractLink end
struct LogLink <: AbstractLink end
struct InverseLink <: AbstractLink end
struct NegativeBinomialLink{T <: AbstractFloat} <: AbstractLink
  Alpha::T
end

# abstract type AbstractProbabilityLink <: AbstractLink end
struct LogitLink <: AbstractLink end
struct ProbitLink <: AbstractLink end
struct CauchitLink <: AbstractLink end
struct OddsPowerLink{T <: AbstractFloat} <: AbstractLink 
  Alpha::T
end
struct LogComplimentLink <: AbstractLink end
struct LogLogLink <: AbstractLink end
struct ComplementaryLogLogLink <: AbstractLink end
struct PowerLink{T <: AbstractFloat} <: AbstractLink
  Alpha::T
end

#=
  Types for Distribution Functions
=#
abstract type AbstractDistribution end
# abstract type AbstractProbabilityDistribution <: AbstractDistribution end
struct BernoulliDistribution <: AbstractDistribution end
struct BinomialDistribution <: AbstractDistribution end
struct GammaDistribution <: AbstractDistribution end
struct PoissonDistribution <: AbstractDistribution end
struct GaussianDistribution <: AbstractDistribution end
struct InverseGaussianDistribution <: AbstractDistribution end
#= Remove this replace with the actual NegativeBinomialDistribution =#
# abstract type AbstractNegativeBernoulliDistribution <: AbstractDistribution end
#= Rename to NegativeBinomialDistribution =#
struct NegativeBernoulliDistribution{T <: AbstractFloat} <: AbstractDistribution
  Alpha::T
end
#= Remove AbstractPowerDistribution =#
# abstract type AbstractPowerDistribution <: AbstractDistribution end
struct PowerDistribution{T <: AbstractFloat} <: AbstractDistribution
  k::T
end
#= Need to implement this! =#
struct LogNormalDistribution <: AbstractDistribution end

InitType{T} = Tuple{Array{T, 1}, Array{T, 1}, Array{T, 1}} where {T <: AbstractFloat}

#=
  Distribution Implementations
=#

# Default Initializer for all the distributions
@inline function init!(::AbstractDistribution, y::Array{T}, wts::Array{T, 1})::InitType{T} where {T <: AbstractFloat}
  y = y[:, 1]
  return y, y, wts
end

#=
  BinomialDistribution
=#

# Initializer
@inline function init!(::BinomialDistribution, y::Array{T}, wts::Array{T, 1})::InitType{T} where {T <: AbstractFloat}
  if size(y)[2] == 1
    y = y[:, 1]
    if length(wts) == 0
      mu = (y .+ T(0.5))./2
    else
      mu = (wts .* y .+ T(0.5))./(wts .+ T(1))
    end
  elseif size(y)[2] == 2
    n = size(y)[1]
    events = y[:, 1]
    N = events .+ y[:, 2]
    y = [if N[i] != 0; events[i]/N[i]; else T(0) end for i in 1:n]
    if length(wts) != 0
      wts .*= N
    else
      wts = N
    end
    mu = (N .* y .+ T(0.5))./(N .+ T(1))
  else
    error("There was a problem")
  end
  return y, mu, wts
end


# Variance Functions
@inline function variance(::BinomialDistribution, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return mu .* (1 .- mu)
end

# As in R
function y_log_y(y::T, x::T)::T where {T <: AbstractFloat}
  return y != 0 ? y * log(y/x) : 0
end
# Deviance and Chi-Squared for BernoulliDistribution/BinomialDistribution
@inline function devianceResiduals(::BinomialDistribution, mu::Array{T, 1},
            y::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return 2 .* (y_log_y.(y, mu) .+ y_log_y.(1 .- y, 1 .- mu))
end
@inline function devianceResiduals(::BinomialDistribution, mu::Array{T, 1},
            y::Array{T, 1}, wts::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return 2 .* wts .* (y_log_y.(y, mu) .+ y_log_y.(1 .- y, 1 .- mu))
end
@inline function X2(distrib::BinomialDistribution, mu::Array{T, 1}, y::Array{T, 1})::T where {T <: AbstractFloat}
  return sum(((y .- mu).^2)./variance(distrib, mu))
end

#=
  PoissonDistribution
=#

@inline function init!(::PoissonDistribution, y::Array{T}, wts::Array{T, 1})::InitType{T} where {T <: AbstractFloat}
  y = y[:, 1]
  mu = y .+ 0.1
  return y, mu, wts
end

# The Variance Function
@inline function variance(::PoissonDistribution, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return mu
end
# Deviance and Chi-Squared for PoissonDistribution
@inline function devianceResiduals(::PoissonDistribution, mu::Array{T, 1}, 
    y::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  dev::Array{T, 1} = zeros(T, size(y))
  for i in 1:length(y)
    if y[i] == 0
      dev[i] = 2 * mu[i]
    else y[i] > 0
      dev[i] = 2 * (y[i] * log(y[i]/mu[i]) - (y[i] - mu[i]))
    end
  end
  return dev
end
@inline function devianceResiduals(::PoissonDistribution, mu::Array{T, 1}, 
    y::Array{T, 1}, wts::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  dev::Array{T, 1} = zeros(T, size(y))
  for i in 1:length(y)
    if y[i] == 0
      dev[i] = 2 * wts[i] * mu[i]
    else y[i] > 0
      dev[i] = 2 * wts[i] * (y[i] * log(y[i]/mu[i]) - (y[i] - mu[i]))
    end
  end
  return dev
end
@inline function X2(distrib::PoissonDistribution, mu::Array{T, 1}, y::Array{T, 1})::T where {T <: AbstractFloat}
  return sum(((y .- mu).^2)./variance(distrib, mu))
end

#=
  GammaDistribution
=#
@inline function variance(::GammaDistribution, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return mu.^2
end
# Deviance and Chi-Squared for GammaDistribution
@inline function devianceResiduals(::GammaDistribution, mu::Array{T, 1}, 
    y::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return 2 .*(((y .- mu)./mu) .- log.(y./mu))
end
@inline function devianceResiduals(::GammaDistribution, mu::Array{T, 1}, 
    y::Array{T, 1}, wts::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return 2 .* wts .*(((y .- mu)./mu) .- log.(y./mu))
end
@inline function X2(distrib::GammaDistribution, mu::Array{T, 1}, y::Array{T, 1})::T where {T <: AbstractFloat}
  return sum(((y .- mu).^2)./variance(distrib, mu))
end


#=
  GaussianDistribution
=#
@inline function variance(::GaussianDistribution, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return fill(T(1), size(mu))
end
# Deviance and Chi-Squared for GaussianDistribution
@inline function devianceResiduals(distrib::GaussianDistribution, mu::Array{T, 1}, 
    y::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return (y .- mu).^2
end
@inline function devianceResiduals(distrib::GaussianDistribution, mu::Array{T, 1}, 
  y::Array{T, 1}, wts::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return wts .* (y .- mu).^2
end
@inline function X2(distrib::GaussianDistribution, mu::Array{T, 1}, y::Array{T, 1})::T where {T <: AbstractFloat} 
  return sum(((y .- mu).^2)./variance(distrib, mu))
end

#=
  InverseGaussianDistribution
=#
@inline function variance(::InverseGaussianDistribution, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return mu.^3
end

# Deviance and Chi-Squared for InverseGaussianDistribution
@inline function devianceResiduals(::InverseGaussianDistribution, mu::Array{T, 1}, 
            y::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return ((y .- mu).^2)./(y .* (mu.^2))
end
@inline function devianceResiduals(::InverseGaussianDistribution, mu::Array{T, 1}, 
    y::Array{T, 1}, wts::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return wts .* ((y .- mu).^2)./(y .* (mu.^2))
end
@inline function X2(distrib::InverseGaussianDistribution, mu::Array{T, 1}, y::Array{T, 1})::T where {T <: AbstractFloat}
  sqDiff = (y .- mu).^2
  v = sqrt(sum(sqDiff)/length(y))
  return v*sum((sqDiff)./variance(distrib, mu))
end

#=
  NegativeBernoulliDistribution
=#
@inline function init!(::NegativeBernoulliDistribution{T}, y::Array{T}, wts::Array{T, 1})::InitType{T} where {T <: AbstractFloat}
  y = y[:, 1]
  mu = copy(y)
  tmp = 1/6
  for i in 1:length(y)
    if y[i] == 0
      mu[i] += tmp
    end
  end
  return y, mu, wts
end

# Variance functions
@inline function variance(distrib::NegativeBernoulliDistribution{T}, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return mu .+ distrib.Alpha .* mu.^2
end

# Deviance and Chi-Squared for NegativeBernoulliDistribution
@inline function devianceResiduals(distrib::NegativeBernoulliDistribution{T}, mu::Array{T, 1}, 
            y::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  Alpha = distrib.Alpha
  dev::Array{T, 1} = zeros(T, size(y))
  iAlpha = T(1)/Alpha
  for i in 1:length(y)
    if y[i] == 0
      dev[i] = 2 * iAlpha*log(1/(1 + Alpha*mu[i]))
    else y[i] > 0
      dev[i] = 2 * (y[i] * log(y[i]/mu[i]) - (y[i] + iAlpha)*log((1 + Alpha*y[i])/(1 + Alpha*mu[i])))
    end
  end
  return dev
end
@inline function devianceResiduals(distrib::NegativeBernoulliDistribution{T}, mu::Array{T, 1}, 
            y::Array{T, 1}, wts::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  Alpha = distrib.Alpha
  dev::Array{T, 1} = zeros(T, size(y))
  iAlpha = T(1)/Alpha
  for i in 1:length(y)
    if y[i] == 0
      dev[i] = 2 * wts[i] * iAlpha*log(1/(1 + Alpha*mu[i]))
    else y[i] > 0
      dev[i] = 2 * wts[i] * (y[i] * log(y[i]/mu[i]) - (y[i] + iAlpha)*log((1 + Alpha*y[i])/(1 + Alpha*mu[i])))
    end
  end
  return dev
end
@inline function X2(distrib::NegativeBernoulliDistribution{T}, mu::Array{T, 1}, y::Array{T, 1})::T where {T <: AbstractFloat}
  return sum(((y .- mu).^2)./variance(distrib, mu))
end


#=
  PowerDistribution
=#

# Variance distribution functions
@inline function variance(distrib::PowerDistribution{T}, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  k = distrib.k
  return mu .^k
end

# Deviance and Chi-Squared for PowerDistribution
@inline function devianceResiduals(distrib::PowerDistribution{T}, mu::Array{T, 1}, 
            y::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  k = distrib.k
  dev::Array{T, 1} = zeros(T, size(y))
  ok = 1 - k
  tk = 2 - k
  for i in 1:length(y)
    dev[i] = ((2*y[i]/(ok*((y[i]^(ok)) - (mu[i]^ok)))) - (2/(tk * ((y[i]^(tk)) - (mu[i]^tk)))))
  end
  return dev
end
@inline function devianceResiduals(distrib::PowerDistribution{T}, mu::Array{T, 1}, 
            y::Array{T, 1}, wts::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  k = distrib.k
  dev::Array{T, 1} = zeros(T, size(y))
  ok = 1 - k
  tk = 2 - k
  for i in 1:length(y)
    dev[i] = wts[i] * ((2*y[i]/(ok*((y[i]^(ok)) - (mu[i]^ok)))) - (2/(tk * ((y[i]^(tk)) - (mu[i]^tk)))))
  end
  return dev
end
@inline function deviance(distrib::AbstractDistribution, mu::Array{T, 1}, y::Array{T, 1})::T where {T <: AbstractFloat}
  return sum(devianceResiduals(distrib, mu, y))
end


#=
  Implementation of the groups of link functions
=#

# IdentityLink Functions
@inline function linkfun(::IdentityLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return copy(mu)
end
@inline function deta_dmu(::IdentityLink, mu::Array{T, 1}, eta::Array{T})::Array{T, 1} where {T <: AbstractFloat}
  return fill(T(1.0), size(mu))
end
@inline function linkinv(::IdentityLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return copy(eta)
end
# This implementation of calculation of z is the same for all the link functions
@inline function Z(link::AbstractLink, y::Array{T, 1}, mu::Array{T, 1}, 
          eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return (deta_dmu(link, mu, eta) .* (y .- mu)) .+ eta
end
#
@inline function W(distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return 1 ./( (deta_dmu(link, mu, eta).^2) .* variance(distrib, mu))
end
# LogLink Functions
@inline function linkfun(::LogLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return log.(mu)
end
@inline function deta_dmu(::LogLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return mu.^(-1)
end
@inline function linkinv(::LogLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return exp.(eta)
end
# InverseLink Functions
@inline function linkfun(::InverseLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return mu.^(-1)
end
@inline function deta_dmu(::InverseLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return -(mu.^-2)
end
@inline function linkinv(::InverseLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return eta.^(-1)
end
# LogitLink Functions
@inline function linkfun(::LogitLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return log.(mu ./(1 .- mu))
end
@inline function deta_dmu(::LogitLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return (mu .* (1 .- mu)).^(-1)
end
@inline function linkinv(::LogitLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  tmp = exp.(eta)
  return tmp ./ (1 .+ tmp)
end

# CauchitLink Functions
@inline function linkfun(::CauchitLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return tan.(pi .* (mu .- (1/2)))
end
@inline function deta_dmu(::CauchitLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return pi .* (cos.(pi .* (mu .- 0.5)).^(-2))
end
@inline function linkinv(::CauchitLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return (atan.(eta)./pi) .+ 0.5
end
# ProbitLink Functions
@inline function linkfun(::ProbitLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return quantile.(standardNormal, mu)
end
@inline function deta_dmu(::ProbitLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return pdf.(standardNormal, eta).^-1
end
@inline function linkinv(::ProbitLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return cdf.(standardNormal, eta)
end

# PowerLink{Alpha} Functions
@inline function linkfun(link::PowerLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  if link.Alpha == T(0)
    return log.(mu)
  end
  return mu .^link.Alpha
end
@inline function deta_dmu(link::PowerLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  if link.Alpha == T(0)
    return mu.^(-1)
  end
  return link.Alpha .* (mu .^(link.Alpha - 1))
end
@inline function linkinv(link::PowerLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  if link.Alpha == T(0)
    return exp.(eta)
  end
  return eta .^(link.Alpha^-1)
end

# OddsPowerLink Functions
#=
  Reference from:
  1. Stata Documentation "rglm - Robust variance estimates for generalized linear models"
  2. Multivariate Methods in Epidemiology by Theodore Holford, Appendix 5 , Table A5-1
=#
# OddsPowerLink Functions
@inline function linkfun(link::OddsPowerLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  if link.Alpha == 0
    return linkfun(LogitLink(), mu)
  end
  return ((mu ./ (1 .- mu)).^link.Alpha .- 1)./link.Alpha
end
@inline function deta_dmu(link::OddsPowerLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  if link.Alpha == 0
    return deta_dmu(LogitLink(), mu, eta)
  end
  return (mu .^ (link.Alpha - 1))./((1 .- mu).^(link.Alpha + 1))
end
@inline function linkinv(link::OddsPowerLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  if link.Alpha == 0
    return linkinv(LogitLink(), eta)
  end
  tmp = ((eta .* link.Alpha .+ 1).^(1/link.Alpha))
  return min.(max.(tmp ./ (1 .+ tmp), eps(T)), ceps(T))
end

# LogComplement Link Functions - needs further work
@inline function linkfun(::LogComplimentLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return log.(max.(1 .- mu, eps(T)))
end
@inline function deta_dmu(::LogComplimentLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return -(max.(1 .- mu, eps(T))).^(-1)
end
@inline function linkinv(::LogComplimentLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return min.(max.(-expm1.(eta), eps(T)), ceps(T))
end
# LogLogLink Link Functions - needs further work
@inline function linkfun(::LogLogLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return -log.(-log.(mu))
end
@inline function deta_dmu(::LogLogLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return -1 ./(mu .* log.(mu))
end
@inline function linkinv(::LogLogLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return exp.(-exp.(-eta))
end
# ComplementaryLogLogLink Functions
@inline function linkfun(::ComplementaryLogLogLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return log.(-log.(1 .- mu))
end
@inline function deta_dmu(::ComplementaryLogLogLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return ((mu .- 1) .* log.(1 .- mu)).^-1
end
@inline function linkinv(::ComplementaryLogLogLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return 1 .- exp.(-exp.(eta))
end
# NegativeBinomialLink Functions - rewrite with 1/Alpha
@inline function linkfun(link::NegativeBinomialLink, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  iAlpha = T(1)/link.Alpha
  return log.(mu ./(mu .+ iAlpha))
end
@inline function deta_dmu(link::NegativeBinomialLink, mu::Array{T, 1}, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return 1 ./ (mu .+ link.Alpha .* mu .^2)
end
@inline function linkinv(link::NegativeBinomialLink, eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  tmp = exp.(eta)
  return tmp ./(link.Alpha .* (1 .- tmp))
end

# Function updates mu and eta - not currenty in use, should 
#          probably use explicit references
function updatemu!(distrib::AbstractDistribution, link::AbstractLink, 
            x::Array{T, 2}, y::Array{T, 1}, eta::Array{T, 1}, 
            mu::Array{T, 1}, coef::Array{T, 1}, offset::Array{T, 1}, 
            residuals::Array{T, 1}, weights::Array{T, 1}, doOffset::Bool, 
            doWeights::Bool)::T where {T <: AbstractFloat}
  eta .= x * coef
  if doOffset
    eta .+= offset
  end
  mu .= linkinv(link, eta)
  if length(weights) == 0
    residuals .= devianceResiduals(distrib, mu, y)
  else
    residuals .= devianceResiduals(distrib, mu, y, weights)
  end
  # if doWeights
  #   residuals .*= weights
  # end
  dev = sum(residuals)
  return dev
end

# For parameters for GLM
struct Control{T <: AbstractFloat}
  epsilon::T
  maxit::Int64
  printError::Bool
  printCoef::Bool
  minstep::T
  function Control{T}(;epsilon::T = T(1E-7), maxit::Int64 = 25, 
                    printError::Bool = false, printCoef::Bool = false,
                    minstep::T = T(1E-5)) where {T <: AbstractFloat}
    return new{T}(epsilon, maxit, printError, printCoef, minstep)
  end
end

# Error functions
using LinearAlgebra: norm
@inline function absoluteError(x::Array{T, 1}, y::Array{T, 1}) where {T <: AbstractFloat}
  return norm(x .- y)
end
@inline function absoluteError(x::T, y::T) where {T <: AbstractFloat}
  return abs(x - y)
end
@inline function relativeError(x::Array{T, 1}, y::Array{T, 1}) where {T <: AbstractFloat}
  return norm(x .- y)/(1E-5 + norm(x))
end
@inline function relativeError(x::T, y::T) where {T <: AbstractFloat}
  return abs(x - y)/(0.1 + abs(x))
end

#=
GLM Class Object
=#
struct GLM{T <: AbstractFloat}
  link::AbstractLink
  distrib::AbstractDistribution
  coefficients::Array{T, 1}
  covariance::Array{T, 2}
  iterations::Int64
  relativeError::T
  absoluteError::T
  converged::Bool
  deviance::T
  residuals::Array{T, 1}
  function GLM(link::AbstractLink, distrib::AbstractDistribution, 
                coefficients::Array{T, 1}, 
                covariance::Array{T, 2}, iterations::Int64,
                relativeError::T, absoluteError::T,
                converged::Bool, deviance::T,
                residuals::Array{T, 1}) where {T <: AbstractFloat}
  return new{T}(link, distrib, coefficients, covariance, iterations, 
              relativeError, absoluteError, converged, deviance, residuals)
  end
end

#=
  Show Method for GLMs
=#
function Base.show(io::IO, ::MIME"text/plain", model::GLM{T}) where {T <: AbstractFloat}
  rep::String = "GLM(" * string(model.link) * ", " * string(model.distrib) * ")\n"
  rep *= "Info(Convergence = " * string(model.converged) * ", " *
          "Iterations = " * string(model.iterations) * ")\n"
  rep *= "Error(AbsoluteError = " * string(model.absoluteError) *
         ", RelativeError = " * string(model.relativeError) * 
         ", Deviance = " * string(model.deviance) * ")\n"
  rep *= "Coefficients:\n" * string(model.coefficients) * "\n"
  standardError = [model.covariance[i, i]^0.5 for i in 1:size(model.covariance)[1]]
  rep *= "Standard Error:\n" * string(standardError) * "\n"
  print(io, rep)
end


function glm(x::Array{T, 2}, y::Array{T}, distrib::AbstractDistribution, link::AbstractLink; 
              offset::Array{T, 1} = Array{T, 1}(undef, 0), weights = Array{T, 1}(undef, 0), 
              control::Control{T} = Control{T}()) where {T <: AbstractFloat}
  
  y, mu, weights = init!(distrib, y, weights)
  eta = linkfun(link, mu)

  coef = zeros(T, size(x)[2])
  coefold = zeros(T, size(x)[2])

  absErr::T = T(Inf)
  relErr::T = T(Inf)

  residuals::Array{T, 1} = zeros(T, size(y))
  dev::T = T(Inf)
  devold::T = T(Inf)

  iter::Int64 = 1

  n, p = size(x)
  converged::Bool = false
  badBreak::Bool = false

  doOffset = false
  if length(offset) != 0
    doOffset = true
  end
  doWeights = false
  if length(weights) != 0
    doWeights = true
  end

  Cov::Array{T, 2} = zeros(T, (p, p))
  while (relErr > control.epsilon)
    if control.printError
      println("Iteration: $iter")
    end
    z::Array{T, 1} = Z(link, y, mu, eta)
    if doOffset
      z .-= offset
    end
    w::Array{T, 1} = W(distrib, link, mu, eta)
    if doWeights
      w .*= weights
    end

    xw::Array{T, 2} = copy(x)
    for j in 1:p
      for i in 1:n
        xw[i, j] = xw[i, j] * w[i]
      end
    end

    Cov = inv(transpose(xw) * x)
    coef::Array{T, 1} = Cov * transpose(xw) * z

    if(control.printCoef)
      println(coef)
    end

    eta = x * coef
    if doOffset
      eta .+= offset
    end
    mu = linkinv(link, eta)
    
    if length(weights) == 0
      residuals = devianceResiduals(distrib, mu, y)
    else
      residuals = devianceResiduals(distrib, mu, y, weights)
    end

    dev = sum(residuals)

    absErr = absoluteError(dev, devold)
    relErr = relativeError(dev, devold)
    # absErr = absoluteError(coef, coefold)
    # relErr = relativeError(coef, coefold)

    frac::T = T(1)
    coefdiff = coef .- coefold
    # Step control
    while (dev > (devold + control.epsilon*dev))
      
      if control.printError
        println("\tStep control")
        println("\tFraction: $frac")
        println("\tDeviance: $dev")
        println("\tAbsolute Error: $absErr")
        println("\tRelative Error: $relErr")
      end

      frac *= 0.5
      coef = coefold .+ (coefdiff .* frac)

      if(control.printCoef)
        println(coef)
      end

      # Abstract this block away into a function so 
      # that it doesn't need to be repeated
      eta = x * coef
      if doOffset
        eta .+= offset
      end
      mu = linkinv(link, eta)

      if length(weights) == 0
        residuals = devianceResiduals(distrib, mu, y)
      else
        residuals = devianceResiduals(distrib, mu, y, weights)
      end

      dev = sum(residuals)
      
      absErr = absoluteError(dev, devold)
      relErr = relativeError(dev, devold)

      if frac < control.minstep
        error("Step control exceeded.")
      end
    end

    devold = dev
    coefold = coef

    if control.printError
      # println("Iteration: $iter")
      println("Deviance: $dev")
      println("Absolute Error: $absErr")
      println("Relative Error: $relErr")
    end

    if iter >= control.maxit
      println("Maximum number of iterations " * string(control.maxit) * " has been reached.")
      badBreak = true
      break
    end

    iter += 1

  end

  if badBreak
    converged = false
  else
    converged = true
  end

  return GLM(link, distrib, coef, Cov, iter, relErr, absErr, converged, 
             dev, residuals)
end

