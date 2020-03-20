# path = "/home/chib/code/glmSolver/"
# include(path * "/julia/GLMSolver.jl")
# using .GLMSolver

"""
  Sampling from a poisson distribution
"""
abstract type AbstractPoissonDistribution <: AbstractDistribution end
struct PoissonDistribution{T} <: AbstractPoissonDistribution
  lambda::T
  function PoissonDistribution(lambda::T) where {T <: AbstractFloat}
    return new{T}(lambda)
  end
end

"""
  Sampling from Poisson Distribution

  Reference:
  Poisson Random Variate Generation, Appl. Statis. (1991), 
      40, No. 1, pp 143 - 158.
  
  # Example:
  sample(PoissonDistribution(1.0), (20,))

"""
function sample(distrib::PoissonDistribution{T}, shape::Tuple{Vararg{Int64}}, version::Val{1}) where {T <: AbstractFloat}
  
  n = prod(shape)
  p = exp(-distrib.lambda)
  ret = zeros(T, shape)

  for i in 1:n
    s = T(1); x = T(0)
    while true
      u = rand(T, 1)[1]
      s *= u
      if s < p
        break
      end
      x += 1
    end
    ret[i] = x
  end
  
  return ret
end

function sample(distrib::PoissonDistribution{T}, shape::Int64, version::Val{1}) where {T <: AbstractFloat}
  return sample(distrib, (shape,))
end


"""
  Function to sample from poisson from a lambda array
"""
function sample(::Type{<: AbstractPoissonDistribution}, lambda::Array{T}, version::Val{1}) where {T <: AbstractFloat}
  
  shape = size(lambda)
  n = prod(shape)
  ret = zeros(T, shape)

  for i in 1:n
    p = exp(-lambda[i])
    s = T(1); x = T(0)
    while true
      u = rand(T, 1)[1]
      s *= u
      if s < p
        break
      end
      x += 1
    end
    ret[i] = x
  end

  return ret
end

function sample(distrib::PoissonDistribution{T}, shape::Tuple{Vararg{Int64}}, version::Val{2}) where {T <: AbstractFloat}
  
  n = prod(shape)
  ret = zeros(T, shape)

  for i in 1:n
    p = exp(-distrib.lambda)
    F = p; x = T(0)
    u = rand(T, 1)[1]
    while true
      if u < F
        break
      end
      x += 1
      p *= (distrib.lambda/x)
      F += p
    end
    ret[i] = x
  end
  
  return ret
end

function sample(distrib::PoissonDistribution{T}, shape::Int64, version::Val{2}) where {T <: AbstractFloat}
  return sample(distrib, (shape,))
end

"""
  Function to sample from poisson from a lambda array
"""
function sample(::Type{<: AbstractPoissonDistribution}, lambda::Array{T}, version::Val{2}) where {T <: AbstractFloat}
  
  shape = size(lambda)
  n = prod(shape)
  ret = zeros(T, shape)
  
  for i in 1:n
    p = exp(-lambda[i])
    F = p; x = T(0)
    u = rand(T, 1)[1]
    while true
      if u < F
        break
      end
      x += 1
      p *= (lambda[i]/x)
      F += p
    end
    ret[i] = x
  end
  
  return ret
end


"""
# Example simulate poisson distribution

  X, eta = simulateData(Float64, 10, 1000);
  y = linkinv(LogLink(), eta);
  y = sample(AbstractPoissonDistribution, y, Val{2}());
"""


"""
  Function to simulate data
"""
function simulateData(::Type{T}, distrib::AbstractDistribution, 
              link::AbstractLink, p::Int64, n::Int64)
  
  X, eta, b = simulateData(T, p, n)
  y = linkinv(link, eta)

  if distrib <: PoissonDistribution
    y = sample(AbstractPoissonDistribution, y, Val{2}())
  end

  if distrib <: BinomialDistribution
    y = map((x) -> T(1)*(x > 0), eta)
  end
    
  return X, y
end
