using LinearAlgebra: eigen, cholesky, diag, diagm, UpperTriangular, LowerTriangular

"""
  Functions for mvrnorm, covriance to correlation conversion (cov2cor)
  and vice versa cor2cov.
"""
function generateRandomMatrix(p::Int64, n::Int64)
  x = rand(n, p)
  return x' * x
end

"""
  Function to convert covariance to correlation matrix
"""
function cov2cor(E::Array{T, 2}) where {T <: AbstractFloat}
  p = size(E)[1]
  d::Array{T, 1} = diag(E)
  d .^= -0.5
  C = (d * d') .* E
  C = 0.5.*(UpperTriangular(C) + LowerTriangular(C)')
  C = C + C'
  for i in 1:p
    C[i, i] = T(1)
  end
  return C
end

"""
  Function to convert correlation to covariance matrix
  d is the variance vector of the covariance matrix
"""
function cor2cov(C::Array{T, 2}, d::Array{T, 1}) where {T <: AbstractFloat}
  p = size(C)[1]
  d1 = d .^0.5
  E = (d1 * d1') .* C
  E = 0.5.*(UpperTriangular(E) + LowerTriangular(E)')
  E = E + E'
  for i in 1:p
    E[i, i] /= 2
  end
  return E
end

"""
  Function that carries samples from a multivariate normal
  distribution when supplied with the number of samples (n),
  the mean vector (mu), and the covariance matrix (sigma).

  Example:
  mvrnorm(100, zeros(10), cov2cor(generateRandomMatrix(10, 10000)))
"""
function mvrnorm(n::Int64, mu::Array{T, 1}, sigma::Array{T, 2}) where {T <: AbstractFloat}
  p = size(sigma)[1]
  A = cholesky(sigma).L
  output = zeros(T, (n, p))
  for i = 1:n
    output[i, :] = mu + A * randn(p)
  end
  return output
end

#=======================================================================================================#
"""
  Random Number Generator for Beta (and Uniform) Distribution
"""
abstract type AbstractSampleDistribution end
struct BetaSampleDistribution{T} <: AbstractSampleDistribution
  alpha::T
  beta::T
  function BetaSampleDistribution(alpha::T, beta::T) where {T <: AbstractFloat}
    return new{T}(alpha, beta)
  end
  function BetaSampleDistribution(alpha::T, beta::T) where {T <: Integer}
    return BetaSampleDistribution(Float64(alpha), Float64(beta))
  end
end

function calcSample(ualpha::T, vbeta::T) where {T <: AbstractFloat}
  return ualpha/(ualpha + vbeta)
end
function condition(ualpha::T, vbeta::T) where {T <: AbstractFloat}
  return (ualpha + vbeta) > T(1)
end


"""
  Version 1 of the random sample from beta distribution
  Reference: C. P. Robert, G. Casella, 
            Monte Carlo Statistical Methods, Example 2.11 p44
"""
function sample(distrib::BetaSampleDistribution{T}, shape::Tuple{Vararg{Int64}}) where {T <: AbstractFloat}
  n = prod(shape)
  U = rand(T, n); V = rand(T, n)
  Y::Array{T} = zeros(T, shape)
  ialpha = 1/distrib.alpha; ibeta = 1/distrib.beta
  for i in 1:n
    u::T = U[i]; v::T = V[i]
    ualpha = u .^ialpha; vbeta = v .^ibeta;
    while condition(ualpha, vbeta)
      u = rand(T, 1)[1]; v = rand(T, 1)[1]
      ualpha = u .^ialpha; vbeta = v .^ibeta;
    end
    Y[i] = calcSample(ualpha, vbeta)
  end
  return Y
end

function sample(distrib::BetaSampleDistribution{T}, shape::Int64) where {T <: AbstractFloat}
  return sample(distrib, (shape,))
end

#=======================================================================================================#

struct UniformSampleDistribution{T} <: AbstractSampleDistribution
  min::T
  max::T
  function UniformSampleDistribution(min::T, max::T) where {T <: AbstractFloat}
    @assert(min < max, "Minimum value is not less than maximum value")
    return new{T}(min, max)
  end
  function UniformSampleDistribution(min::T, max::T) where {T <: Integer}
    return UniformSampleDistribution(Float64(min), Float64(max))
  end
end

function sample(distrib::UniformSampleDistribution{T}, shape::Tuple{Vararg{Int64}}) where {T <: AbstractFloat}
  rsample::Array{T} = rand(T, shape)
  return (rsample .* (distrib.max - distrib.min)) .+ distrib.min
end
function sample(distrib::UniformSampleDistribution{T}, shape::Int64) where {T <: AbstractFloat}
  return sample(distrib, (shape,))
end


function Base.min(x::Array{T}) where {T <: AbstractFloat}
  n = length(x)
  @assert(n > 0, "Length of array is zero.")
  if n == 1
    return x[1]
  end
  ret::T = x[1]
  for i in 2:n
    ret = ret > x[i] ? x[i] : ret
  end
  return ret
end

function Base.max(x::Array{T}) where {T <: AbstractFloat}
  n = length(x)
  @assert(n > 0, "Length of array is zero.")
  if n == 1
    return x[1]
  end
  ret::T = x[1]
  for i in 2:n
    ret = ret < x[i] ? x[i] : ret
  end
  return ret
end

function Base.range(x::Array{T}) where {T <: AbstractFloat}
  return [min(x), max(x)]
end


#=======================================================================================================#

"""
  Identity Matrix
"""
function I(::Type{T}, p::Int64) where {T <: Number}
  x = zeros(T, (p, p))
  for i in 1:p
    x[i, i] = T(1)
  end
  return x
end
"""
  Convenience overload
"""
function I(p::Int64)
  return I(Float64, p)
end

"""
  Types for generating random correlation matrices
"""
abstract type AbstractRandomCorrelationMatrix end

struct BetaGenerator <: AbstractRandomCorrelationMatrix end
struct OnionGenerator <: AbstractRandomCorrelationMatrix end
struct UniformGenerator <: AbstractRandomCorrelationMatrix end
struct VineGenerator <: AbstractRandomCorrelationMatrix end

"""
  Vine method for generating a random correlation matrix
  Source:
    https://stats.stackexchange.com/questions/2746/how-to-efficiently-generate-random-positive-semidefinite-correlation-matrices/125017#125017
    
    Check it with to make sure the solutions are correct:
    Journal of Multivariate Analysis 100 (2009) 1989–2001
    Generating random correlation matrices based on vines and extended
    onion method. Daniel Lewandowski(a)*, Dorota Kurowicka (a), 
    Harry Joe (b).
  
  randomCorrelationMatrix(VineGenerator(), 10, 2.0)
"""
function randomCorrelationMatrix(::VineGenerator, d::Int64, eta::T) where {T <: AbstractFloat}
  
  beta = eta + (d - 1)/2
  P = zeros(T, (d, d))
  S = I(T, d)
  
  for k in 1:(d - 1)
    beta = beta - (1/2)
    distrib = BetaSampleDistribution(beta, beta)
  
    for i in (k + 1):d
      P[k, i] = sample(distrib, 1)[1]
      P[k, i] = (P[k, i] - 0.5)*2
      p = P[k, i]
      
      for l in (k - 1):-1:1
        p = p * sqrt((1 - P[l, i]^2)*(1 - P[l, k]^2)) + P[l, i]*P[l, k]
      end
      
      S[k, i] = p
      S[i, k] = p
    end
  end
  
  return S
end

"""
  Random correlation matrix using the Onion Generator
"""
function randomCorrelationMatrix(::OnionGenerator, d::Int64, eta::T) where {T <: AbstractFloat}
  beta::T = eta + (d - 2)/2
  distrib = BetaSampleDistribution(beta, beta)
  u = sample(distrib, 1)[1]
  r = I(T, 2)
  r[1, 2] = r[2, 1] = 2*u - 1
  for k in 2:(d - 1)
    beta -= T(1/2)
    distrib = BetaSampleDistribution(T(k/2), beta)
    y = sample(distrib, 1)[1]
    U = rand(T, k)
    w = sqrt(y) .* U
    
    ev = eigen(r)
    A = ev.vectors * diagm(abs.(ev.values).^(0.5)) * ev.vectors'
    z = A * w
    r = [r z; z' T(1)]
  end
  return r
end

"""
  Random correlation matrix by sampling from Beta Distribution
  # Example:
  randomCorrelationMatrix(BetaGenerator(), 10, (1.0, 1.0))
"""
function randomCorrelationMatrix(::BetaGenerator, d::Int64, (alpha, beta)::Tuple{T, T}) where {T <: AbstractFloat}
  distrib = BetaSampleDistribution(alpha, beta)
  r = sample(distrib, (d, d))
  # Change range to (-1, 1)
  r .= (r .* 2) .+ T(-1)
  # Symmetry
  r .= T(0.5) .* (r + r')
  for i in 1:d
    r[i, i] = T(1)
  end
  ev = eigen(r)
  r = ev.vectors * diagm(sort(abs.(ev.values))) * ev.vectors'
  maxR = max(r)
  r ./= maxR
  r .= T(0.5) .* (r + r')
  for i in 1:d
    r[i, i] = T(1)
  end
  return r
end

"""
  Random correlation matrix by sampling from Uniform Distribution
  # Example:
  randomCorrelationMatrix(Float64, UniformGenerator(), 10)
"""
function randomCorrelationMatrix(::Type{T}, ::UniformGenerator, d::Int64) where {T <: AbstractFloat}
  distrib = UniformSampleDistribution(T(-1), T(1))
  r = sample(distrib, (d, d))
  r .= T(0.5) .* (r + r')
  for i in 1:d
    r[i, i] = T(1)
  end
  ev = eigen(r)
  r = ev.vectors * diagm(sort(abs.(ev.values))) * ev.vectors'
  maxR = max(abs.(r)...)
  r ./= maxR
  r .= T(0.5) .* (r + r')
  for i in 1:d
    r[i, i] = T(1)
  end
  return r
end

#=======================================================================================================#

using Statistics: mean

"""
  Function to simulate X and eta, p = number of parameters 
  (including intercept), n = number of samples.

  # Example:
    using Random: seed!
    seed!(0)
    simulateData(Float64, 10, 1000)
"""
function simulateData(::Type{T}, p::Int64, n::Int64, delta::T = T(0)) where {T <: AbstractFloat}
  
  corr = randomCorrelationMatrix(T, UniformGenerator(), p)
  mu = zeros(T, p)
  X = mvrnorm(n, mu, corr)

  #= The intercept =#
  X[:, 1] .= T(1)
  
  b = zeros(T, p)
  idist = UniformSampleDistribution(T(0), T(0.3))
  b[1] = sample(idist, 1)[1]
  if length(b) > 1
    distrib = UniformSampleDistribution(T(-0.1), T(0.1))
    b[2:p] = sample(distrib, p - 1)
  end
  
  eta = X*b
  sd = 0.5*abs(mean(eta))
  eta = delta .+ eta .+ sd .* randn(n)

  return (X = X, eta = eta)
end

