#=
  GLM Link Functions
=#

using Distributions

abstract type AbstractMatrixType end
struct RegularData <: AbstractMatrixType end
struct Block1D <: AbstractMatrixType end
struct Block1DParallel <: AbstractMatrixType end

# Defining the standard normal for further use
const standardNormal = Normal()

# Standard Normal pdf
function pdfSN(eta::T)::T where {T <: AbstractFloat}
  return exp(-(eta^2)/2)/sqrt(2*pi)
end

# Compliment of epsilon 1 - eps(Float)
function ceps(T::Type{<: AbstractFloat})
  return 1 - eps(T)
end

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

#====================================================================
  Copy constructors for link types
====================================================================#
import Base.copy
copy(::IdentityLink) = IdentityLink()
copy(::LogLink) = LogLink()
copy(::InverseLink) = InverseLink()
function copy(distrib::NegativeBinomialLink{T}) where {T}
  return NegativeBinomialLink(distrib.Alpha)
end
copy(::LogitLink) = LogitLink()
copy(::ProbitLink) = ProbitLink()
copy(::CauchitLink) = CauchitLink()
function copy(distrib::OddsPowerLink{T}) where {T}
  return OddsPowerLink(distrib.Alpha)
end
copy(::LogComplimentLink) = LogComplimentLink()
copy(::LogLogLink) = LogLogLink()
copy(::ComplementaryLogLogLink) = ComplementaryLogLogLink()
function copy(distrib::PowerLink{T}) where {T}
  return PowerLink(distrib.Alpha)
end
#===================================================================#


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


# For block algorithm
function linkfun(link::AbstractLink, mu::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [linkfun(link, mu[i]) for i in 1:nBlocks]
end
function deta_dmu(link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [deta_dmu(link, mu[i], eta[i]) for i in 1:nBlocks]
end
function linkinv(link::AbstractLink, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(eta)
  return [linkinv(link, eta[i]) for i in 1:nBlocks]
end

# For parallel block algorithm
function linkfun(::Block1DParallel, link::AbstractLink, mu::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = linkfun(link, mu[i])
  end
  return ret
end
function deta_dmu(::Block1DParallel, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = deta_dmu(link, mu[i], eta[i])
  end
  return ret
end
function linkinv(::Block1DParallel, link::AbstractLink, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(eta)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = linkinv(link, eta[i])
  end
  return ret
end
