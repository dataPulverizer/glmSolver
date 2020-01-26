#=
  GLM Family Functions
=#

#=
  Types for Distribution Functions
=#
abstract type AbstractDistribution end
struct BernoulliDistribution <: AbstractDistribution end
struct BinomialDistribution <: AbstractDistribution end
struct GammaDistribution <: AbstractDistribution end
struct PoissonDistribution <: AbstractDistribution end
struct GaussianDistribution <: AbstractDistribution end
struct InverseGaussianDistribution <: AbstractDistribution end
struct NegativeBernoulliDistribution{T <: AbstractFloat} <: AbstractDistribution
  Alpha::T
end
struct PowerDistribution{T <: AbstractFloat} <: AbstractDistribution
  k::T
end
# struct LogNormalDistribution <: AbstractDistribution end


# Type returned from init!() function
InitType{T} = Tuple{Array{T, 1}, Array{T, 1}, Array{T, 1}} where {T <: AbstractFloat}

#=
  Distribution Implementations
=#

# Default Initializer for all the distributions
@inline function init!(::AbstractDistribution, y::Array{T}, wts::Array{T, 1})::InitType{T} where {T <: AbstractFloat}
  y = y[:, 1]
  return y, y, wts
end

function init(::AbstractDistribution, y::Array{Array{T}, 1}, wts::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(y)
  if length(size(y[1])) == 2
    y = [y[i][:, 1] for i in 1:nBlocks]
  end
  mu = Array{Array{T, 1}, 1}(undef, nBlocks);
  for i in 1:nBlocks
    mu[i] = copy(y[i]);
  end
  return y, mu, wts;
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

@inline function init!(::BinomialDistribution, y::Array{Array{T}, 1}, wts::Array{Array{T, 1}, 1})::InitType{Array{T, 1}} where {T <: AbstractFloat}
  nBlocks::Int64 = length(y)
  if size(y[1]) == 1
    y = [y[i][:, 1] for i in 1:nBlocks]
    if length(wts) == 0
      mu = [(y[i] .+ T(0.5))./2 for i in 1:nBlocks]
    else
      mu = [(wts[i] .* y .+ T(0.5))./(wts[i] .+ T(1)) for i in 1:nBlocks]
    end
  elseif size(y[1])[2] == 2
    events = [y[i][:, 1] for i in 1:nBlocks]
    N = [events[i] .+ y[i][:, 2] for i in 1:nBlocks]
    for i in 1:nBlocks
      n = size(y[i])[1]
      tmp = zeros(T, n)
      for j in 1:n
        tmp[j] = if N[i][j] != 0; events[i][j]/N[i][j]; else T(0) end
      end
      y[i] = tmp
    end
    # y = [if N[i] != 0; events[i]/N[i]; else T(0) end for i in 1:n]
    if length(wts) != 0
      [wts[i] .*= N[i] for i in 1:nBlocks]
    else
      wts = N
    end
    mu = [(N[i] .* y[i] .+ T(0.5))./(N[i] .+ T(1)) for i in 1:nBlocks]
  else
    error("There was a problem with the dimensions of y")
  end
  return y, mu, wts
end


# Variance Functions
@inline function variance(::BinomialDistribution, mu::Array{T, 1})::Array{T, 1} where {T <: AbstractFloat}
  return mu .* (1 .- mu)
end

function variance(distrib::AbstractDistribution, mu::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [variance(distrib, mu[i]) for i in 1:nBlocks]
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

function devianceResiduals(distrib::AbstractDistribution, mu::Array{Array{T, 1}, 1},
        y::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks = length(y)
  return [devianceResiduals(distrib, mu[i], y[i]) for i in 1:nBlocks]
end

function devianceResiduals(distrib::AbstractDistribution, mu::Array{Array{T, 1}, 1},
        y::Array{Array{T, 1}, 1}, wts::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks = length(y)
  return [devianceResiduals(distrib, mu[i], y[i], wts[i]) for i in 1:nBlocks]
end


#=
  PoissonDistribution
=#

@inline function init!(::PoissonDistribution, y::Array{T}, wts::Array{T, 1})::InitType{T} where {T <: AbstractFloat}
  y = y[:, 1]
  mu = y .+ T(0.1)
  return y, mu, wts
end

function init!(::PoissonDistribution, y::Array{Array{T}, 1},
            wts::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(y)
  tmp = T(0.1)
  y = [y[i][:, 1] for i in 1:nBlocks]
  mu = [y[i][:, 1] .+ tmp for i in 1:nBlocks]
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
  tmp = T(1/6)
  for i in 1:length(y)
    if y[i] == 0
      mu[i] += tmp
    end
  end
  return y, mu, wts
end

function init!(::NegativeBernoulliDistribution{T}, y::Array{Array{T}, 1},
            wts::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(y)
  tmp = T(1/6)
  y = [y[i][:, 1] for i in 1:nBlocks]
  mu = [y[i][:, 1] .+ tmp for i in 1:nBlocks]
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
