#=
  Solvers for GLM
=#
abstract type AbstractSolver end

#=
  GLM Solver uses (X'WX)^(-1) * X'Wy
=#
struct VanillaSolver <: AbstractSolver end

#=
  Solver Using QR Decomposition
=#
struct QRSolver <: AbstractSolver end

#=
  TODO:
  1. Unify the QRSolve and Vanilla solve interface as in
     the D implementation.
=#
function solve(::Type{<: VanillaSolver}, X::Array{T, 2}, y::Array{T, 1}) where {T <: AbstractFloat}
  ret::Array{T, 1} = Array{T, 1}(undef, 0)
  try
    ret = (X'*X)/(X'*y)
  catch
    ret = inv(X'*X)*(X'*y)
  end
  return ret
  #return inv(X'*X)*(X'*y)
end

import LinearAlgebra.LAPACK.geqrf!
import LinearAlgebra.LAPACK.orgqr!
import LinearAlgebra.BLAS.gemv
import LinearAlgebra.LAPACK.trtrs!

# import LinearAlgebra.UpperTriangular
# import LinearAlgebra.LAPACK.trtri!
# import LinearAlgebra.BLAS.trsm!


#=
  The QR Solver
=#
function getUpperR(QR::Array{T, 2}) where {T <: AbstractFloat}
  _, p = size(QR)
  R = zeros(T, (p, p))
  for i in 1:p
    for j in 1:i
      R[j, i] = QR[j, i]
    end
  end
  return R
end

function solve(::Type{<: QRSolver}, Q::Array{T, 2}, y::Array{T, 1}) where {T <: AbstractFloat}
  Q = copy(Q)
  _, p = size(Q)
  tau = zeros(T, p)
  geqrf!(Q, tau)
  R = getUpperR(Q)
  orgqr!(Q, tau, p)
  z = gemv('T', 1.0, Q, y)
  # z = Q' * y
  trtrs!('U', 'N', 'N', R, z)
  return z
end

#============================  Matrix Inverses ============================#
#= Matrix Inverse By LU Decomposition =#
import LinearAlgebra.LAPACK.getrf!
import LinearAlgebra.LAPACK.getri!


#=
  Inverse Types
=#
abstract type AbstractInverse end

#= GETRI Matrix Inverse =#
struct GETRIInverse <: AbstractInverse end
#= POTRI Matrix Inverse =#
struct POTRIInverse <: AbstractInverse end
#= SYTRF Matrix Inverse =#
struct SYTRFInverse <: AbstractInverse end
#= GESVD Matrix Inverse =#
struct GESVDInverse <: AbstractInverse end

#=
  Inverse of a square matrix using the GETRI function
  uses LU decomposition. The geqrf() does the QR decomposition
  and the getri() function carries out an inverse from the output
  of geqrf().
=#
function inv(::GETRIInverse, A::Array{T, 2}) where {T <: AbstractFloat}
  A = copy(A)
  p, _ = size(A)
  A, ipiv, info = getrf!(A)
  @assert(info == 0, "getrf! error in argument " * string(info))
  getri!(A, ipiv)
  return A
end

#= Matrix Inverse By Cholesky Decomposition For Positive Definite Matrices =#
import LinearAlgebra.LAPACK.potrf!
import LinearAlgebra.LAPACK.potri!

#=
  # Examples:
  x = rand(100, 10);
  x = x' * x
  sum(abs.(Base.inv(x) .- inv(POTRIInverse(), x)))
=#
function inv(::POTRIInverse, A::Array{T, 2}) where {T <: AbstractFloat}
  p, _ = size(A)
  A = copy(A)
  A, info = potrf!('U', A)
  @assert(info == 0, "potrf error in argument " * string(info))
  potri!('U', A)
  for i in 1:p
    for j in 1:i
      A[i, j] = A[j, i]
    end
  end
  return A
end

#= Matrix Inverse By LU Decomposition For Symmetric Indefinite Matrices =#
import LinearAlgebra.LAPACK.sytri!# (uplo, A, ipiv)
import LinearAlgebra.LAPACK.sytrf!# (uplo, A) -> (A, ipiv, info)

#=
  # Examples:
  x = rand(100, 10);
  x = x' * x
  sum(abs.(Base.inv(x) .- inv(SYTRFInverse(), x)))
=#
function inv(::SYTRFInverse, A::Array{T, 2}) where {T <: AbstractFloat}
  p, _ = size(A)
  A = copy(A)
  A, ipiv, info = sytrf!('U', A)
  sytri!('U', A, ipiv)
  for i in 1:p
    for j in 1:i
      A[i, j] = A[j, i]
    end
  end
  return A
end

#= Generalized Matrix Inverse By SVD  =#
import LinearAlgebra.LAPACK.gesvd! # (jobu, jobvt, A) -> (U, S, VT)

#= Matrix Vector Sweep Function =#
function multSweep(xw::Array{T, 2}, w::Array{T, 1})::Array{T, 2} where {T <: AbstractFloat}
  n, p = size(xw)
  for j in 1:p
    for i in 1:n
      xw[i, j] = xw[i, j] * w[i]
    end
  end
  return xw
end

#=
  # Examples:
  x = rand(100, 10);
  x = x' * x
  sum(abs.(Base.inv(x) .- inv(GESVDInverse(), x)))
=#
function inv(::GESVDInverse, A::Array{T, 2}) where {T <: AbstractFloat}
  p, _ = size(A)
  A = copy(A)
  U, S, Vt = gesvd!('A', 'A', A)
  S = [el > 1E-9 ? 1/el : el for el in S]
  return multSweep(Vt, S)' * U'
end

#============================== Linear Equation Solvers ==============================#
#= For Ax = b =#
#==============================  GESV Solver ==============================#

function calcXWX(xwx::Array{T, 2}, x::Array{T, 2}, w::Array{T, 1}, z::Array{T, 1}) where {T <: AbstractFloat}
  # n, p = size(x)
  xw = copy(x)
  xw = multSweep(xw, w)
  xwx = xw' * x
  xwz = xw' * z
  return xwx, xwz
end

function calcXWX(xwx::Array{T, 2}, x::Array{Array{T, 2}, 1}, w::Array{Array{T, 1}, 1}, z::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  p::Int64 = size(x[1])[2]
  nBlocks::Int64 = length(x)
  xwx = zeros(T, (p, p))
  xwz = zeros(T, (p, 1))
  for i in 1:nBlocks
    xw = copy(x[i])
    xw = multSweep(xw, w[i])
    xwx += xw' * x[i]
    xwz += xw' * z[i]
  end
  return xwx, xwz
end

function calcXWX(::Block1DParallel, xwx::Array{T, 2}, x::Array{Array{T, 2}, 1}, w::Array{Array{T, 1}, 1}, z::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  p::Int64 = size(x[1])[2]
  nBlocks::Int64 = length(x)
  xwx = [zeros(T, (p, p)) for i in 1:nthreads()]
  xwz = [zeros(T, (p, 1)) for i in 1:nthreads()]
  @threads for i in 1:nBlocks
    xw = copy(x[i])
    xw = multSweep(xw, w[i])
    xwx[threadid()] += xw' * x[i]
    xwz[threadid()] += xw' * z[i]
  end
  for i in 2:nthreads()
    xwx[1] .+= xwx[i]
    xwz[1] .+= xwz[i]
  end
  return xwx[1], xwz[1]
end

function calcXW(xwx::Array{T, 2}, x::Array{T, 2}, w::Array{T, 1}, z::Array{T, 1}) where {T <: AbstractFloat}
  # n, p = size(x)
  xw = copy(x)
  xw = multSweep(xw, w)
  return xw, z .* w
end


struct GESVSolver <: AbstractSolver end
# Weights for the GESVSolver
@inline function W(::GESVSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-1)
end
function W(::GESVSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-1) for i in 1:nBlocks]
end
function W(matrixType::Block1DParallel, ::GESVSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = ((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-1)
  end
  return ret
end

# Solver
import LinearAlgebra.LAPACK.gesv!
function solve!(::GESVSolver, R::Array{T, 2}, 
    xwx::Array{T, 2}, xw::Array{T, 2},
    x::Array{T, 2}, z::Array{T, 1}, w::Array{T, 1},
    coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(xwx, x, w, z)
  coef, _, ipiv = gesv!(copy(xwx), xwz)
  return (R, xwx, xw, x, z, w, coef)
end
function solve!(::GESVSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{Array{T, 2},1},
  x::Array{Array{T, 2}, 1}, z::Array{Array{T, 1}, 1},
  w::Array{Array{T, 1}, 1}, coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(xwx, x, w, z)
  coef, _, ipiv = gesv!(copy(xwx), xwz)
  return (R, xwx, xw, x, z, w, coef)
end
function solve!(matrixType::Block1DParallel, ::GESVSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{Array{T, 2},1},
  x::Array{Array{T, 2}, 1}, z::Array{Array{T, 1}, 1},
  w::Array{Array{T, 1}, 1}, coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(matrixType, xwx, x, w, z)
  coef, _, ipiv = gesv!(copy(xwx), xwz)
  return (R, xwx, xw, x, z, w, coef)
end
# Default Covariance Function
function cov(::AbstractSolver, invType::AbstractInverse, R::Array{T, 2}, xwx::Array{T, 2}, xw::Array{T, 2})::Array{T, 2} where {T <: AbstractFloat}
  return inv(invType, xwx)
end

function cov(::AbstractSolver, invType::AbstractInverse, R::Array{T, 2}, xwx::Array{T, 2})::Array{T, 2} where {T <: AbstractFloat}
  return inv(invType, xwx)
end

#==============================  POSV Solver ==============================#
#= Solver For Positive Definite Matrices =#
# Weights for posv!(uplo, A, B) -> (A, B)
import LinearAlgebra.LAPACK.posv! # (uplo, A, B) -> (A, B)

struct POSVSolver <: AbstractSolver end
# Weights for the POSVSolver
@inline function W(::POSVSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-1)
end
function W(::POSVSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-1) for i in 1:nBlocks]
end
function W(matrixType::Block1DParallel, ::POSVSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = ((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-1)
  end
  return ret
end
# Solver
function solve!(::POSVSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{T, 2},
  x::Array{T, 2}, z::Array{T, 1}, w::Array{T, 1},
  coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(xwx, x, w, z)
  _, coef = posv!('U', copy(xwx), xwz)
  return (R, xwx, xw, x, z, w, coef)
end
function solve!(::POSVSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{Array{T, 2},1},
  x::Array{Array{T, 2}, 1}, z::Array{Array{T, 1}, 1},
  w::Array{Array{T, 1}, 1}, coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(xwx, x, w, z)
  _, coef = posv!('U', copy(xwx), xwz)
  return (R, xwx, xw, x, z, w, coef)
end
function solve!(matrixType::Block1DParallel, ::POSVSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{Array{T, 2},1},
  x::Array{Array{T, 2}, 1}, z::Array{Array{T, 1}, 1},
  w::Array{Array{T, 1}, 1}, coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(matrixType, xwx, x, w, z)
  _, coef = posv!('U', copy(xwx), xwz)
  return (R, xwx, xw, x, z, w, coef)
end
#==============================  SYSV Solver ==============================#
#= Solver For Symmetric Indefinite Matrices =#
import LinearAlgebra.LAPACK.sysv! # (uplo, A, B) -> (B, A, ipiv)

struct SYSVSolver <: AbstractSolver end
# Weights for the SYSVSolver
@inline function W(::SYSVSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-1)
end
function W(::SYSVSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-1) for i in 1:nBlocks]
end
function W(matrixType::Block1DParallel, ::SYSVSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = ((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-1)
  end
  return ret
end
# Solver
function solve!(::SYSVSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{T, 2},
  x::Array{T, 2}, z::Array{T, 1}, w::Array{T, 1},
  coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(xwx, x, w, z)
  coef, _, _ = sysv!('U', copy(xwx), xwz)
  return (R, xwx, xw, x, z, w, coef)
end
function solve!(::SYSVSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{Array{T, 2},1},
  x::Array{Array{T, 2}, 1}, z::Array{Array{T, 1}, 1},
  w::Array{Array{T, 1}, 1}, coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(xwx, x, w, z)
  coef, _, _ = sysv!('U', copy(xwx), xwz)
  return (R, xwx, xw, x, z, w, coef)
end
function solve!(matrixType::Block1DParallel, ::SYSVSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{Array{T, 2},1},
  x::Array{Array{T, 2}, 1}, z::Array{Array{T, 1}, 1},
  w::Array{Array{T, 1}, 1}, coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(matrixType, xwx, x, w, z)
  coef, _, _ = sysv!('U', copy(xwx), xwz)
  return (R, xwx, xw, x, z, w, coef)
end
#============================== Least Squares Solvers ==============================#
#==============================  GELS Solver ==============================#
#= Solver For Symmetric Indefinite Matrices =#
import LinearAlgebra.LAPACK.gels! # (trans, A, B) -> (F, B, ssr)

struct GELSSolver <: AbstractSolver end
# Weights for the GELSSolver
@inline function W(::GELSSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-0.5)
end
function W(::GELSSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-0.5) for i in 1:nBlocks]
end
function W(matrixType::Block1DParallel, ::GELSSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = ((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-0.5)
  end
  return ret
end

# Solver
function solve!(::GELSSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{T, 2},
  x::Array{T, 2}, z::Array{T, 1}, w::Array{T, 1},
  coef::Array{T, 1}) where {T <: AbstractFloat}
  _, p = size(x)
  xw, zw = calcXW(xwx, x, w, z)
  F, coef, ssr = gels!('N', xw, zw)
  for j in 1:p
    for i in 1:j
      if i <= j
        R[i, j] = xw[i, j]
      end
    end
  end
  return (R, xwx, xw, x, z, w, coef)
end
function cov(::GELSSolver, invType::AbstractInverse, R::Array{T, 2}, xwx::Array{T, 2}, xw::Array{T, 2})::Array{T, 2} where {T <: AbstractFloat}
  xwx = R' * R
  return inv(invType, xwx)
end
#==============================  GELSY Solver ==============================#
import LinearAlgebra.LAPACK.gelsy! # (A, B, rcond) -> (B, rnk)
struct GELSYSolver <: AbstractSolver end
# Weights for the GELSYSolver
@inline function W(::GELSYSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-0.5)
end
function W(::GELSYSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-0.5) for i in 1:nBlocks]
end
function W(::GELSYSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = ((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-0.5)
  end
  return ret
end

# Solver
function solve!(::GELSYSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{T, 2},
  x::Array{T, 2}, z::Array{T, 1}, w::Array{T, 1},
  coef::Array{T, 1}) where {T <: AbstractFloat}
  _, p = size(x); rcond = T(0)
  xw, zw = calcXW(xwx, x, w, z)
  coef, rnk = gelsy!(copy(xw), zw, rcond)
  return (R, xwx, xw, x, z, w, coef)
end
function cov(::GELSYSolver, invType::AbstractInverse, R::Array{T, 2}, xwx::Array{T, 2}, xw::Array{T, 2})::Array{T, 2} where {T <: AbstractFloat}
  xwx = xw' * xw
  return inv(invType, xwx)
end
#==============================  GELSD Solver ==============================#
#= Solver For Symmetric Indefinite Matrices =#
import LinearAlgebra.LAPACK.gelsd! # (A, B, rcond) -> (B, rnk)

struct GELSDSolver <: AbstractSolver end
# Weights for the GELSDSolver
@inline function W(::GELSDSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-0.5)
end
function W(::GELSDSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-0.5) for i in 1:nBlocks]
end
# Solver
function solve!(::GELSDSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{T, 2},
  x::Array{T, 2}, z::Array{T, 1}, w::Array{T, 1},
  coef::Array{T, 1}) where {T <: AbstractFloat}
  _, p = size(x)
  rcond::T = T(0)
  xw, zw = calcXW(xwx, x, w, z)
  coef, _ = gelsd!(copy(xw), zw, rcond)
  return (R, xwx, xw, x, z, w, coef)
end
function cov(::GELSDSolver, invType::AbstractInverse, R::Array{T, 2}, xwx::Array{T, 2}, xw::Array{T, 2})::Array{T, 2} where {T <: AbstractFloat}
  xwx = xw' * xw
  return inv(invType, xwx)
end

#==============================  GRADIENT DESCENT SOLVERS ==============================#
#=======================================================================================#

function XWX(x::Array{T, 2}, w::Array{T, 1}) where {T <: AbstractFloat}
  xw = copy(x)
  xw = multSweep(xw, w)
  xwx = xw' * x
  return xwx
end

function XWX(x::Array{Array{T, 2}, 1}, w::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  p::Int64 = size(x[1])[2]
  nBlocks::Int64 = length(x)
  xwx = zeros(T, (p, p))
  for i in 1:nBlocks
    xw = copy(x[i])
    xw = multSweep(xw, w[i])
    xwx += xw' * x[i]
  end
  return xwx
end

function XWX(::Block1DParallel, x::Array{Array{T, 2}, 1}, w::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  p::Int64 = size(x[1])[2]
  nBlocks::Int64 = length(x)
  xwx = [zeros(T, (p, p)) for i in 1:nthreads()]
  @threads for i in 1:nBlocks
    xw = copy(x[i])
    xw = multSweep(xw, w[i])
    xwx[threadid()] += xw' * x[i]
  end
  for i in 2:nthreads()
    xwx[1] .+= xwx[i]
  end
  return xwx[1]
end


abstract type AbstractGradientDescentSolver end

# Weights for the GradientDescentSolver
@inline function W(::AbstractGradientDescentSolver, 
            distrib::AbstractDistribution, link::AbstractLink, 
            mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-1)
end
function W(::AbstractGradientDescentSolver, distrib::AbstractDistribution,
            link::AbstractLink, mu::Array{Array{T, 1}, 1}, 
            eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  return [((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-1) for i in 1:nBlocks]
end
function W(matrixType::Block1DParallel, ::AbstractGradientDescentSolver, 
            distrib::AbstractDistribution, link::AbstractLink, 
            mu::Array{Array{T, 1}, 1}, 
            eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  nBlocks::Int64 = length(mu)
  ret::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    ret[i] = ((deta_dmu(link, mu[i], eta[i]).^2) .* variance(distrib, mu[i])).^(-1)
  end
  return ret
end

# GD Covariance Function
function cov(::AbstractGradientDescentSolver, invType::AbstractInverse, xwx::Array{T, 2})::Array{T, 2} where {T <: AbstractFloat}
  return inv(invType, xwx)
end

#==============================  GRADIENTS ==============================#

#=
  Internal gradient function
=#
function _gradient(distrib::AbstractDistribution, link::AbstractLink,
            y::Array{T, 1}, x::Array{T, 2}, mu::Array{T, 1},
            eta::Array{T, 1}) where {T <: AbstractFloat}
  n, p = size(x)
  grad = zeros(T, p)
  numer::Array{T, 1} = y .- mu
  X2::T = sum((numer.^2) ./ variance(distrib, mu))
  tmp::Array{T, 1} = numer ./(deta_dmu(link, mu, eta) .* variance(distrib, mu))
  for j in 1:p
    grad[j] += sum(tmp .* x[:, j])
  end
  return (X2 = X2, grad = grad)
end

#=
  Gradient function for regular data
=#
function gradient(distrib::AbstractDistribution, link::AbstractLink,
            y::Array{T, 1}, x::Array{T, 2}, mu::Array{T, 1},
            eta::Array{T, 1}) where {T <: AbstractFloat}
  n, p = size(x)
  tmp = _gradient(distrib, link, y, x, mu, eta)
  df = n - p
  @assert(df > 0, "Number of items not greater than number of parameters")
  phi::T = tmp.X2/df
  return tmp.grad ./phi
end

#=
  Gradient function for blocked data
=#
function gradient(distrib::AbstractDistribution, link::AbstractLink, 
  y::Array{Array{T, 1}, 1}, x::Array{Array{T, 2}, 1},
  mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  
  nBlocks::Int64 = length(y); n::Int64 = T(0)
  p::Int64 = size(x[1])[2]; grad::Array{T, 1} = zeros(T, p)
  X2::T = T(0)

  for i in 1:nBlocks
    n += length(y[i])
    tmp = _gradient(distrib, link, y[i], x[i], mu[i], eta[i])
    # println("size(grad): ", size(grad))
    # println("size(tmp.grad): ", size(tmp.grad))
    grad .+= tmp.grad
    X2 += tmp.X2
  end

  df = n - p
  phi::T = X2/df
  @assert(df > 0, "Number of items not greater than number of parameters")
  return grad ./phi
end


function gradient(::Block1DParallel, distrib::AbstractDistribution, link::AbstractLink, 
  y::Array{Array{T, 1}, 1}, x::Array{Array{T, 2}, 1},
  mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}) where {T <: AbstractFloat}
  
  nBlocks::Int64 = length(y); p::Int64 = size(x[1])[2]; n::Int64 = T(0);
  # grad::Array{T, 1} = zeros(T, p); X2::T = T(0)
  
  nStore = [Int64(0) for i in 1:nthreads()]
  X2Store = [T(0) for i in 1:nthreads()]
  gradStore = [zeros(T, p) for i in 1:nthreads()]
  
  @threads for i in 1:nBlocks
    tmp = _gradient(distrib, link, y[i], x[i], mu[i], eta[i])
    gradStore[threadid()] .+= tmp.grad
    X2Store[threadid()] += tmp.X2
    nStore[threadid()] += length(y[i])
  end
  
  for i in 2:nthreads()
    nStore[1] += nStore[i]
    gradStore[1] .+= gradStore[i]
    X2Store[1] += X2Store[i]
  end
  
  df = nStore[1] - p
  phi::T = X2Store[1]/df
  @assert(df > 0, "Number of items not greater than number of parameters")
  
  return gradStore[1]./phi
end

#==============================  Simple Gradient Descent Solver ==============================#

struct GradientDescentSolver{T} <: AbstractGradientDescentSolver
  learningRate::T
  function GradientDescentSolver(learningRate::T = T(1E-6)) where {T <: AbstractFloat}
    return new{T}(learningRate)
  end
end

function solve!(solver::GradientDescentSolver, distrib::AbstractDistribution, 
  link::AbstractLink, y::Array{T, 1}, x::Array{T, 2}, mu::Array{T, 1},
  eta::Array{T, 1}, coef::Array{T, 1}) where {T <: AbstractFloat}
  
  grad = gradient(distrib, link, y, x, mu, eta);
  coef .+= solver.learningRate .* grad;
  return coef
end

function solve!(solver::GradientDescentSolver, distrib::AbstractDistribution,
  link::AbstractLink, y::Array{Array{T, 1}, 1}, x::Array{Array{T, 2}, 1},
  mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}, coef::Array{T, 1}) where {T <: AbstractFloat}
  
  grad = gradient(distrib, link, y, x, mu, eta);
  coef .+= solver.learningRate .* grad;
  return coef
end

function solve!(::Block1DParallel, solver::GradientDescentSolver, 
  distrib::AbstractDistribution, link::AbstractLink, 
  y::Array{Array{T, 1}, 1}, x::Array{Array{T, 2}, 1},
  mu::Array{Array{T, 1}, 1}, eta::Array{Array{T, 1}, 1}, coef::Array{T, 1}) where {T <: AbstractFloat}
  
  grad = gradient(distrib, link, y, x, mu, eta);
  coef .+= solver.learningRate .* grad;
  return coef
end
