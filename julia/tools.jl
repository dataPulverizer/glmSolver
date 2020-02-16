#=
  Error Functions, GLM Object, Z, W
=#

# This implementation of calculation of z is the same for all the link functions
@inline function Z(link::AbstractLink, y::Array{T, 1}, mu::Array{T, 1}, 
            eta::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return (deta_dmu(link, mu, eta) .* (y .- mu)) .+ eta
end

function Z(link::AbstractLink, y::Array{Array{T, 1}, 1}, mu::Array{Array{T, 1}, 1}, 
            eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(y)
  return [(deta_dmu(link, mu[i], eta[i]) .* (y[i] .- mu[i])) .+ eta[i] for i in 1:nBlocks]
end

function Z(::Block1DParallel, link::AbstractLink, y::Array{Array{T, 1}, 1}, mu::Array{Array{T, 1}, 1}, 
            eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(y)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = (deta_dmu(link, mu[i], eta[i]) .* (y[i] .- mu[i])) .+ eta[i]
  end
  return ret
end

# Weights for the VanillaSolver
@inline function W(::VanillaSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-1)
end
function W(::VanillaSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-1) for i in 1:nBlocks]
end
function W(::Block1DParallel, ::VanillaSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = ((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-1)
  end
  return ret
end

# Weights for the QRSolver
@inline function W(::QRSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-0.5)
end
function W(::QRSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-0.5) for i in 1:nBlocks]
end
function W(::Block1DParallel, ::QRSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = ((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-0.5)
  end
  return ret
end

# The actual weights function used
@inline function W(distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ( (deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-0.5)
end
function W(distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [( (deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-0.5) for i in 1:nBlocks]
end
function W(::Block1DParallel, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = ((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-0.5)
  end
  return ret
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

abstract type AbstractGLM end

struct GLM{T <: AbstractFloat} <: AbstractGLM
  link::AbstractLink
  distrib::AbstractDistribution
  phi::T
  coefficients::Array{T, 1}
  covariance::Array{T, 2}
  iterations::Int64
  relativeError::T
  absoluteError::T
  converged::Bool
  deviance::T
  residuals::Array{T, 1}
  function GLM(link::AbstractLink, distrib::AbstractDistribution, 
                phi::T, coefficients::Array{T, 1}, 
                covariance::Array{T, 2}, iterations::Int64,
                relativeError::T, absoluteError::T,
                converged::Bool, deviance::T,
                residuals::Array{T, 1}) where {T <: AbstractFloat}
  return new{T}(link, distrib, phi, coefficients, covariance, iterations, 
              relativeError, absoluteError, converged, deviance, residuals)
  end
end


struct GLMBlock1D{T <: AbstractFloat} <: AbstractGLM
  link::AbstractLink
  distrib::AbstractDistribution
  phi::T
  coefficients::Array{T, 1}
  covariance::Array{T, 2}
  iterations::Int64
  relativeError::T
  absoluteError::T
  converged::Bool
  deviance::T
  residuals::Array{Array{T, 1}, 1}
  function GLMBlock1D(link::AbstractLink, distrib::AbstractDistribution, 
                phi::T, coefficients::Array{T, 1}, 
                covariance::Array{T, 2}, iterations::Int64,
                relativeError::T, absoluteError::T,
                converged::Bool, deviance::T,
                residuals::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  return new{T}(link, distrib, phi, coefficients, covariance, iterations, 
              relativeError, absoluteError, converged, deviance, residuals)
  end
end




#=
  Show Method for GLMs
=#
function Base.show(io::IO, ::MIME"text/plain", model::AbstractGLM)
  rep::String = "GLM(" * string(model.link) * ", " * string(model.distrib) * ")\n"
  rep *= "Info(Convergence = " * string(model.converged) * ", " *
          "Iterations = " * string(model.iterations) * ")\n"
  rep *= "Error(AbsoluteError = " * string(model.absoluteError) *
         ", RelativeError = " * string(model.relativeError) * 
         ", Deviance = " * string(model.deviance) * ", phi = " * string(model.phi) * ")\n"
  rep *= "Coefficients:\n" * string(model.coefficients) * "\n"
  standardError = [model.covariance[i, i]^0.5 for i in 1:size(model.covariance)[1]]
  rep *= "Standard Error:\n" * string(standardError) * "\n"
  print(io, rep)
end
