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
  Fix this later
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






