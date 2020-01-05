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
  x = transpose(x) * x
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
  x = transpose(x) * x
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

#============================== Linear Equation Solvers ==============================#
#= For Ax = b =#
#==============================  GESV Solver ==============================#

function calcXWX(xwx::Array{T, 2}, x::Array{T, 2}, w::Array{T, 1}, z::Array{T, 1}) where {T <: AbstractFloat}
  n, p = size(x)
  xw = copy(x)
  for j in 1:p
    for i in 1:n
      xw[i, j] = xw[i, j] * w[i]
    end
  end
  xwx = transpose(xw) * x
  xwz = transpose(xw) * z
  return xwx, xwz
end

function calcXW(xwx::Array{T, 2}, x::Array{T, 2}, w::Array{T, 1}, z::Array{T, 1}) where {T <: AbstractFloat}
  n, p = size(x)
  xw = copy(x)
  for j in 1:p
    for i in 1:n
      xw[i, j] = xw[i, j] * w[i]
    end
  end
  return xw, z .* w
end


struct GESVSolver <: AbstractSolver end
# Weights for the GESVSolver
@inline function W(::GESVSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-1)
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
# Default Covariance Function
function cov(::AbstractSolver, invType::AbstractInverse, R::Array{T, 2}, xwx::Array{T, 2}, xw::Array{T, 2})::Array{T, 2} where {T <: AbstractFloat}
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
# Solver
function solve!(::POSVSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{T, 2},
  x::Array{T, 2}, z::Array{T, 1}, w::Array{T, 1},
  coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(xwx, x, w, z)
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
# Solver
function solve!(::SYSVSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{T, 2},
  x::Array{T, 2}, z::Array{T, 1}, w::Array{T, 1},
  coef::Array{T, 1}) where {T <: AbstractFloat}
  xwx, xwz = calcXWX(xwx, x, w, z)
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
  xwx = transpose(R) * R
  return inv(invType, xwx)
end
#==============================  GELSY Solver ==============================#
import LinearAlgebra.LAPACK.gelsy! # (A, B, rcond) -> (B, rnk)
struct GELSYSolver <: AbstractSolver end
# Weights for the GELSYSolver
@inline function W(::GELSYSolver, distrib::AbstractDistribution, link::AbstractLink, mu::Array{T, 1}, eta::Array{T, 1}) where {T <: AbstractFloat}
  return ((deta_dmu(link, mu, eta).^2) .* variance(distrib, mu)).^(-0.5)
end
# Solver
function solve!(::GELSYSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{T, 2},
  x::Array{T, 2}, z::Array{T, 1}, w::Array{T, 1},
  coef::Array{T, 1}) where {T <: AbstractFloat}
  _, p = size(x); rcond = T(0)
  xw, zw = calcXW(xwx, x, w, z)
  coef, rnk = gelsy!(xw, zw, rcond)
  return (R, xwx, xw, x, z, w, coef)
end
function cov(::GELSYSolver, invType::AbstractInverse, R::Array{T, 2}, xwx::Array{T, 2}, xw::Array{T, 2})::Array{T, 2} where {T <: AbstractFloat}
  xwx = transpose(xw) * xw
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
# Solver
function solve!(::GELSDSolver, R::Array{T, 2}, 
  xwx::Array{T, 2}, xw::Array{T, 2},
  x::Array{T, 2}, z::Array{T, 1}, w::Array{T, 1},
  coef::Array{T, 1}) where {T <: AbstractFloat}
  _, p = size(x)
  rcond::T = T(0)
  xw, zw = calcXW(xwx, x, w, z)
  coef, _ = gelsd!(xw, zw, rcond)
  return (R, xwx, xw, x, z, w, coef)
end
function cov(::GELSDSolver, invType::AbstractInverse, R::Array{T, 2}, xwx::Array{T, 2}, xw::Array{T, 2})::Array{T, 2} where {T <: AbstractFloat}
  xwx = transpose(xw) * xw
  return inv(invType, xwx)
end

