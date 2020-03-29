#=
  Module To Fit GLM
=#

function glm(::RegularData, x::Array{T, 2}, y::Array{T}, distrib::AbstractDistribution, link::AbstractLink, 
              solver::AbstractSolver; inverse::AbstractInverse = GETRIInverse(),
              offset::Array{T, 1} = Array{T, 1}(undef, 0), 
              weights = Array{T, 1}(undef, 0), control::Control{T} = Control{T}()) where {T <: AbstractFloat}
  
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
  xwx::Array{T, 2} = zeros(T, (p, p))
  R::Array{T, 2} = zeros(T, (p, p))
  xw::Array{T, 2} = zeros(T, (0, 0))
  while (relErr > control.epsilon)
    if control.printError
      println("Iteration: $iter")
    end
    z::Array{T, 1} = Z(link, y, mu, eta)
    if doOffset
      z .-= offset
    end
    
    
    w::Array{T, 1} = W(solver, distrib, link, mu, eta)
    # w::Array{T, 1} = W(distrib, link, mu, eta)
    if doWeights
      w .*= weights
    end

    # xw::Array{T, 2} = copy(x)
    # for j in 1:p
    #   for i in 1:n
    #     xw[i, j] = xw[i, j] * w[i]
    #   end
    # end
    # z .*= w

    # Overwrites xw in the QR case!
    # coef = solve(solver, xw, z)
    
    # coef = solve!(solver, Ref(R), Ref(xwx), Ref(xw), 
    #               Ref(x), Ref(z), Ref(w), Ref(coef))

    R, xwx, xw, x, z, w, coef = solve!(solver, R, xwx, xw, 
                                    x, z, w, coef)
    
    # println("xwx:\n", xwx)
    # println("coef:\n", coef)
    
    if(control.printCoef)
      println(coef)
    end

    eta = x * coef
    if doOffset
      eta .+= offset
    end
    mu = linkinv(link, eta)
    
    # println("Iteration: $iter");

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

    # Cov = xw' * xw
    Cov = cov(solver, inverse, R, xwx, xw)

    if control.printError
      # println("Iteration: $iter")
      println("Deviance: $dev")
      println("Absolute Error: $absErr")
      println("Relative Error: $relErr")
    end

    if iter >= control.maxit
      println("Maximum number of iterations " * 
                string(control.maxit) * " has been reached.")
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

  phi::T = T(1)
  if (typeof(distrib) != BernoulliDistribution) | (typeof(distrib) != BinomialDistribution) | (typeof(distrib) != PoissonDistribution)
    phi = dev/(n - p)
    Cov .*= phi
  end

  return GLM(link, distrib, phi, coef, Cov, iter, relErr, absErr, converged, 
             dev, residuals)
end


#============================== BLOCK GLM FUNCTION ==============================#
#=
  Valid only for solvers: GESVSolver, POSVSolver, SYSVSolver
=#
function glm(::Block1D, x::Array{Array{T, 2}, 1}, y::Array{Array{T, 2}, 1}, distrib::AbstractDistribution, link::AbstractLink, 
              solver::AbstractSolver; inverse::AbstractInverse = GETRIInverse(),
              offset::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, 0), 
              weights = Array{Array{T, 1}, 1}(undef, 0), control::Control{T} = Control{T}()) where {T <: AbstractFloat}
  
  y, mu, weights = init!(distrib, y, weights)
  eta = linkfun(link, mu)

  coef = zeros(T, size(x[1])[2])
  coefold = zeros(T, size(x[1])[2])

  absErr::T = T(Inf)
  relErr::T = T(Inf)

  nBlocks::Int64 = length(mu)

  residuals::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, 0)
  dev::T = T(Inf)
  devold::T = T(Inf)

  iter::Int64 = 1

  n = sum(map((y) -> size(y)[1], x))
  p = size(x[1])[2]
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
  xwx::Array{T, 2} = zeros(T, (p, p))
  R::Array{T, 2} = zeros(T, (p, p))
  xw::Array{Array{T, 2}, 1} = Array{Array{T, 2}, 1}(undef, 0)
  while (relErr > control.epsilon)
    if control.printError
      println("Iteration: $iter")
    end
    z::Array{Array{T, 1}, 1} = Z(link, y, mu, eta)
    if doOffset
      for i in 1:nBlocks
        z[i] .-= offset[i]
      end
    end
    
    
    w::Array{Array{T, 1}, 1} = W(solver, distrib, link, mu, eta)
    if doWeights
      for i in 1:nBlocks
        w[i] .*= weights[i]
      end
    end

    # println("Coefficient: ", coef)
    
    # println("Typeof coeff before: ", typeof(coef))
    R, xwx, xw, x, z, w, coef = solve!(solver, R, xwx, xw, 
                                    x, z, w, coef)
    coef = reshape(coef, (p,))
    
    # println("xwx:\n", xwx)
    # println("coef:\n", coef)
    
    if(control.printCoef)
      println(coef)
    end

    eta = [ (x[i] * coef)[:, 1] for i in 1:nBlocks]
    if doOffset
      for i in 1:nBlocks
        eta[i] .+= offset[i]
      end
    end
    mu = linkinv(link, eta)
    
    if length(weights) == 0
      residuals = devianceResiduals(distrib, mu, y)
    else
      residuals = devianceResiduals(distrib, mu, y, weights)
    end

    dev = T(0)
    for i in 1:nBlocks
      dev += sum(residuals[i])
    end

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
      eta = [ (x[i] * coef)[:, 1] for i in 1:nBlocks]
      if doOffset
        for i in 1:nBlocks
          eta[i] .+= offset[i]
        end
      end
      mu = linkinv(link, eta)

      if length(weights) == 0
        residuals = devianceResiduals(distrib, mu, y)
      else
        residuals = devianceResiduals(distrib, mu, y, weights)
      end

      # dev = sum(residuals)
      dev = T(0)
      for i in 1:nBlocks
        dev += sum(residuals[i])
      end
      
      absErr = absoluteError(dev, devold)
      relErr = relativeError(dev, devold)

      if frac < control.minstep
        error("Step control exceeded.")
      end
    end

    devold = dev
    coefold = coef

    # Cov = xw' * xw
    Cov = cov(solver, inverse, R, xwx)

    if control.printError
      # println("Iteration: $iter")
      println("Deviance: $dev")
      println("Absolute Error: $absErr")
      println("Relative Error: $relErr")
    end

    if iter >= control.maxit
      println("Maximum number of iterations " * 
                string(control.maxit) * " has been reached.")
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

  phi::T = T(1)
  if (typeof(distrib) != BernoulliDistribution) | (typeof(distrib) != BinomialDistribution) | (typeof(distrib) != PoissonDistribution)
    phi = dev/(n - p)
    Cov .*= phi
  end

  return GLMBlock1D(link, distrib, phi, coef, Cov, iter, relErr, absErr, converged, 
             dev, residuals)
end


#============================== BLOCK PARALLEL GLM FUNCTION ==============================#
#=
  Valid only for solvers: GESVSolver, POSVSolver, SYSVSolver
=#
function glm(matrixType::Block1DParallel, x::Array{Array{T, 2}, 1}, y::Array{Array{T, 2}, 1}, distrib::AbstractDistribution, link::AbstractLink, 
              solver::AbstractSolver; inverse::AbstractInverse = GETRIInverse(),
              offset::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, 0), 
              weights = Array{Array{T, 1}, 1}(undef, 0), control::Control{T} = Control{T}()) where {T <: AbstractFloat}
  
  #= Set BLAS threads =#
  set_num_threads(1)
  y, mu, weights = init!(matrixType, distrib, y, weights)
  eta = linkfun(matrixType, link, mu)

  coef = zeros(T, size(x[1])[2])
  coefold = zeros(T, size(x[1])[2])

  absErr::T = T(Inf)
  relErr::T = T(Inf)

  nBlocks::Int64 = length(mu)

  residuals::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, 0)
  dev::T = T(Inf)
  devold::T = T(Inf)

  iter::Int64 = 1

  n = sum(map((y) -> size(y)[1], x))
  p = size(x[1])[2]
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
  xwx::Array{T, 2} = zeros(T, (p, p))
  R::Array{T, 2} = zeros(T, (p, p))
  xw::Array{Array{T, 2}, 1} = Array{Array{T, 2}, 1}(undef, 0)

  while (relErr > control.epsilon)

    if control.printError
      println("Iteration: $iter")
    end
    z::Array{Array{T, 1}, 1} = Z(matrixType, link, y, mu, eta)
    if doOffset
      @threads for i in 1:nBlocks
        z[i] .-= offset[i]
      end
    end
    
    w::Array{Array{T, 1}, 1} = W(matrixType, solver, distrib, link, mu, eta)
    if doWeights
      @threads for i in 1:nBlocks
        w[i] .*= weights[i]
      end
    end

    # println("Coefficient: ", coef)
    
    # println("Typeof coeff before: ", typeof(coef))
    R, xwx, xw, x, z, w, coef = solve!(matrixType, solver, R, 
                                    xwx, xw, x, z, w, coef)
    coef = reshape(coef, (p,))
    
    # println("xwx:\n", xwx)
    # println("coef:\n", coef)
    
    if(control.printCoef)
      println(coef)
    end

    # eta = [ (x[i] * coef)[:, 1] for i in 1:nBlocks]
    @threads for i in 1:nBlocks
      eta[i] = (x[i] * coef)[:, 1]
    end
    if doOffset
      @threads for i in 1:nBlocks
        eta[i] .+= offset[i]
      end
    end
    mu = linkinv(matrixType, link, eta)
    
    if length(weights) == 0
      residuals = devianceResiduals(matrixType, distrib, mu, y)
    else
      residuals = devianceResiduals(matrixType, distrib, mu, y, weights)
    end

    dev = T(0)
    # Parallel Reduction Required
    for i in 1:nBlocks
      dev += sum(residuals[i])
    end

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

      # eta = [ (x[i] * coef)[:, 1] for i in 1:nBlocks]
      @threads for i in 1:nBlocks
        eta[i] = (x[i] * coef)[:, 1]
      end
      if doOffset
        for i in 1:nBlocks
          eta[i] .+= offset[i]
        end
      end
      mu = linkinv(matrixType, link, eta)

      if length(weights) == 0
        residuals = devianceResiduals(matrixType, distrib, mu, y)
      else
        residuals = devianceResiduals(matrixType, distrib, mu, y, weights)
      end

      # dev = sum(residuals)
      dev = T(0)
      # Parallel Reduction Required
      for i in 1:nBlocks
        dev += sum(residuals[i])
      end
      
      absErr = absoluteError(dev, devold)
      relErr = relativeError(dev, devold)

      if frac < control.minstep
        error("Step control exceeded.")
      end
    end

    devold = dev
    coefold = coef

    # Cov = xw' * xw
    Cov = cov(solver, inverse, R, xwx)

    if control.printError
      # println("Iteration: $iter")
      println("Deviance: $dev")
      println("Absolute Error: $absErr")
      println("Relative Error: $relErr")
    end

    if iter >= control.maxit
      println("Maximum number of iterations " * 
                string(control.maxit) * " has been reached.")
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

  phi::T = T(1)
  if (typeof(distrib) != BernoulliDistribution) | (typeof(distrib) != BinomialDistribution) | (typeof(distrib) != PoissonDistribution)
    phi = dev/(n - p)
    Cov .*= phi
  end

  set_num_threads(nthreads())

  return GLMBlock1D(link, distrib, phi, coef, Cov, iter, relErr, absErr, converged, 
             dev, residuals)
end

#============================== GRADIENT DESCENT ==============================#
#==============================================================================#

#============================== REGULAR DATA ==============================#
function glm(matrixType::RegularData, x::Array{T, 2}, y::Array{T},
              distrib::AbstractDistribution, link::AbstractLink,
              solver::AbstractGradientDescentSolver; 
              inverse::AbstractInverse = GETRIInverse(), 
              control::Control{T} = Control{T}(), 
              calculateCovariance::Bool = true, 
              doStepControl::Bool = true, 
              offset::Array{T, 1} = Array{T, 1}(undef, 0), 
              weights = Array{T, 1}(undef, 0)) where {T <: AbstractFloat}
  
  y, weights = init!(solver, distrib, y, weights)
  
  coef = zeros(T, size(x)[2])
  coefold = zeros(T, size(x)[2])

  eta = x * coef
  doOffset = false
  if length(offset) != 0
    doOffset = true
  end

  if doOffset
    eta .+= offset
  end
  mu = linkinv(link, eta)

  absErr::T = T(Inf)
  relErr::T = T(Inf)

  residuals::Array{T, 1} = zeros(T, size(y))
  dev::T = T(Inf)
  devold::T = T(Inf)

  iter::Int64 = 1

  n, p = size(x)
  converged::Bool = false
  badBreak::Bool = false

  doWeights = false
  if length(weights) != 0
    doWeights = true
  end

  vcov::Array{T, 2} = fill(T(1), (p, p))
  xwx::Array{T, 2} = zeros(T, (p, p))
  w::Array{T, 1} = zeros(T, (0,))

  while (relErr > control.epsilon)
  
    if control.printError
      println("Iteration: $iter")
    end

    solver = iteration(solver, iter)

    if typeof(solver) <: NesterovSolver
      coef .= NesterovModifier(solver, coef);
      
      eta = x * coef
      if doOffset
        eta .+= offset
      end
      mu = linkinv(link, eta)

      coef .= NesterovUnModifier(solver, coef);
    end

    solver, coef = solve!(matrixType, solver, distrib, link, y, x, mu, eta, coef)
    # println("Coef gradient descent regular data: ", coef, "\n")
    
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

    frac::T = T(1)
    coefdiff = coef .- coefold
    # Step control
    while (doStepControl && (dev > (devold + control.epsilon*dev)))
      
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
      println("Deviance: $dev")
      println("Absolute Error: $absErr")
      println("Relative Error: $relErr")
    end

    if iter >= control.maxit
      println("Maximum number of iterations " * 
                string(control.maxit) * " has been reached.")
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
  
  phi::T = T(1)
  if calculateCovariance
    w = W(solver, distrib, link, mu, eta)
    if doWeights
      w .*= weights
    end
    xwx = XWX(x, w)
    vcov = cov(solver, inverse, xwx)
    if (typeof(distrib) != BernoulliDistribution) | (typeof(distrib) != BinomialDistribution) | (typeof(distrib) != PoissonDistribution)
      phi = dev/(n - p)
      vcov .*= phi
    end
  end

  return GLM(link, distrib, phi, coef, vcov, iter, relErr, absErr, converged, 
             dev, residuals)
end

#============================== BLOCK DATA GRADIENT DESCENT ==============================#

function glm(matrixType::Block1D, x::Array{Array{T, 2}, 1}, y::Array{Array{T, 2}, 1}, 
              distrib::AbstractDistribution, link::AbstractLink, 
              solver::AbstractGradientDescentSolver; 
              inverse::AbstractInverse = GETRIInverse(),
              control::Control{T} = Control{T}(), 
              calculateCovariance::Bool = true, 
              doStepControl::Bool = true, 
              offset::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, 0), 
              weights = Array{Array{T, 1}, 1}(undef, 0)) where {T <: AbstractFloat}
  
  y, weights = init!(solver, distrib, y, weights)

  coef = zeros(T, size(x[1])[2])
  coefold = zeros(T, size(x[1])[2])

  nBlocks::Int64 = length(y)
  eta = [ (x[i] * coef)[:, 1] for i in 1:nBlocks]
  doOffset = false
  if length(offset) != 0
    doOffset = true
  end

  if doOffset
    for i in 1:nBlocks
      eta[i] .+= offset[i]
    end
  end
  mu = linkinv(link, eta)

  absErr::T = T(Inf)
  relErr::T = T(Inf)

  residuals::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, 0)
  dev::T = T(Inf)
  devold::T = T(Inf)

  iter::Int64 = 1

  n = sum(map((y) -> size(y)[1], x))
  p = size(x[1])[2]
  converged::Bool = false
  badBreak::Bool = false

  doWeights = false
  if length(weights) != 0
    doWeights = true
  end

  vcov::Array{T, 2} = fill(T(1), (p, p))
  xwx::Array{T, 2} = zeros(T, (p, p))
  w::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, 0)

  while (relErr > control.epsilon)
    
    if control.printError
      println("Iteration: $iter")
    end

    solver = iteration(solver, iter)

    if typeof(solver) <: NesterovSolver
      coef .= NesterovModifier(solver, coef);
      
      eta = [ (x[i] * coef)[:, 1] for i in 1:nBlocks]
      if doOffset
        for i in 1:nBlocks
          eta[i] .+= offset[i]
        end
      end
      mu = linkinv(link, eta)

      coef .= NesterovUnModifier(solver, coef);
    end

    solver, coef = solve!(matrixType, solver, distrib, link, y, x, mu, eta, coef)
    coef = reshape(coef, (p,))
    
    if(control.printCoef)
      println(coef)
    end

    eta = [ (x[i] * coef)[:, 1] for i in 1:nBlocks]
    if doOffset
      for i in 1:nBlocks
        eta[i] .+= offset[i]
      end
    end
    mu = linkinv(link, eta)
    
    if length(weights) == 0
      residuals = devianceResiduals(distrib, mu, y)
    else
      residuals = devianceResiduals(distrib, mu, y, weights)
    end

    dev = T(0)
    for i in 1:nBlocks
      dev += sum(residuals[i])
    end

    absErr = absoluteError(dev, devold)
    relErr = relativeError(dev, devold)
    # absErr = absoluteError(coef, coefold)
    # relErr = relativeError(coef, coefold)

    frac::T = T(1)
    coefdiff = coef .- coefold
    # Step control
    while (doStepControl && (dev > (devold + control.epsilon*dev)))
      
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
      eta = [ (x[i] * coef)[:, 1] for i in 1:nBlocks]
      if doOffset
        for i in 1:nBlocks
          eta[i] .+= offset[i]
        end
      end
      mu = linkinv(link, eta)

      if length(weights) == 0
        residuals = devianceResiduals(distrib, mu, y)
      else
        residuals = devianceResiduals(distrib, mu, y, weights)
      end

      dev = T(0)
      for i in 1:nBlocks
        dev += sum(residuals[i])
      end
      
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
      println("Maximum number of iterations " * 
                string(control.maxit) * " has been reached.")
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

  phi::T = T(1)
  if calculateCovariance
    w = W(solver, distrib, link, mu, eta)
    if doWeights
      for i in 1:nBlocks
        w[i] .*= weights[i]
      end
    end
    xwx = XWX(x, w)
    vcov = cov(solver, inverse, xwx)
    if (typeof(distrib) != BernoulliDistribution) | (typeof(distrib) != BinomialDistribution) | (typeof(distrib) != PoissonDistribution)
      phi = dev/(n - p)
      vcov .*= phi
    end
  end

  return GLMBlock1D(link, distrib, phi, coef, vcov, iter, relErr, absErr, converged, 
             dev, residuals)
end

#============================== BLOCK PARALLEL GRADIENT DESCENT ==============================#
#=
  Valid only for solvers: GESVSolver, POSVSolver, SYSVSolver
=#
function glm(matrixType::Block1DParallel, x::Array{Array{T, 2}, 1}, 
              y::Array{Array{T, 2}, 1}, distrib::AbstractDistribution, 
              link::AbstractLink, solver::AbstractGradientDescentSolver; 
              inverse::AbstractInverse = GETRIInverse(), 
              control::Control{T} = Control{T}(), 
              calculateCovariance::Bool = true, 
              doStepControl::Bool = true, 
              offset::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, 0), 
              weights = Array{Array{T, 1}, 1}(undef, 0)) where {T <: AbstractFloat}
  
  #= Set BLAS threads =#
  set_num_threads(1)
  
  y, weights = init!(solver, matrixType, distrib, y, weights)

  coef = zeros(T, size(x[1])[2])
  coefold = zeros(T, size(x[1])[2])

  nBlocks::Int64 = length(y)
  eta::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, nBlocks)
  @threads for i in 1:nBlocks
    eta[i] = (x[i] * coef)[:, 1]
  end

  doOffset = false
  if length(offset) != 0
    doOffset = true
  end
  if doOffset
    @threads for i in 1:nBlocks
      eta[i] .+= offset[i]
    end
  end
  mu = linkinv(matrixType, link, eta)

  absErr::T = T(Inf)
  relErr::T = T(Inf)

  residuals::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, 0)
  dev::T = T(Inf)
  devold::T = T(Inf)

  iter::Int64 = 1

  n = sum(map((y) -> size(y)[1], x))
  p = size(x[1])[2]
  converged::Bool = false
  badBreak::Bool = false

  
  doWeights = false
  if length(weights) != 0
    doWeights = true
  end

  vcov::Array{T, 2} = fill(T(1), (p, p))
  xwx::Array{T, 2} = zeros(T, (p, p))
  w::Array{Array{T, 1}, 1} = Array{Array{T, 1}, 1}(undef, 0)

  while (relErr > control.epsilon)

    if control.printError
      println("Iteration: $iter")
    end

    solver = iteration(solver, iter)

    if typeof(solver) <: NesterovSolver
      coef .= NesterovModifier(solver, coef);
      
      @threads for i in 1:nBlocks
        eta[i] = (x[i] * coef)[:, 1]
      end
      if doOffset
        @threads for i in 1:nBlocks
          eta[i] .+= offset[i]
        end
      end
      mu = linkinv(matrixType, link, eta)

      coef .= NesterovUnModifier(solver, coef);
    end

    # println("Coefficient: ", coef)
    
    solver, coef = solve!(matrixType, solver, distrib, link, y, x, mu, eta, coef)
    coef = reshape(coef, (p,))
    
    if(control.printCoef)
      println(coef)
    end

    @threads for i in 1:nBlocks
      eta[i] = (x[i] * coef)[:, 1]
    end
    if doOffset
      @threads for i in 1:nBlocks
        eta[i] .+= offset[i]
      end
    end
    mu = linkinv(matrixType, link, eta)
    
    if length(weights) == 0
      residuals = devianceResiduals(matrixType, distrib, mu, y)
    else
      residuals = devianceResiduals(matrixType, distrib, mu, y, weights)
    end

    dev = T(0)
    # Parallel Reduction Required
    for i in 1:nBlocks
      dev += sum(residuals[i])
    end

    absErr = absoluteError(dev, devold)
    relErr = relativeError(dev, devold)
    # absErr = absoluteError(coef, coefold)
    # relErr = relativeError(coef, coefold)

    frac::T = T(1)
    coefdiff = coef .- coefold
    # Step control
    while (doStepControl && (dev > (devold + control.epsilon*dev)))
      
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

      # eta = [ (x[i] * coef)[:, 1] for i in 1:nBlocks]
      @threads for i in 1:nBlocks
        eta[i] = (x[i] * coef)[:, 1]
      end
      if doOffset
        for i in 1:nBlocks
          eta[i] .+= offset[i]
        end
      end
      mu = linkinv(matrixType, link, eta)

      if length(weights) == 0
        residuals = devianceResiduals(matrixType, distrib, mu, y)
      else
        residuals = devianceResiduals(matrixType, distrib, mu, y, weights)
      end

      dev = T(0)
      # Parallel Reduction Required
      for i in 1:nBlocks
        dev += sum(residuals[i])
      end
      
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
      println("Maximum number of iterations " * 
                string(control.maxit) * " has been reached.")
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

  phi::T = T(1)
  if calculateCovariance
    w = W(matrixType, solver, distrib, link, mu, eta)
    if doWeights
      @threads for i in 1:nBlocks
        w[i] .*= weights[i]
      end
    end
    xwx = XWX(x, w)
    vcov = cov(solver, inverse, xwx)
    if (typeof(distrib) != BernoulliDistribution) | (typeof(distrib) != BinomialDistribution) | (typeof(distrib) != PoissonDistribution)
      phi = dev/(n - p)
      vcov .*= phi
    end
  end

  set_num_threads(nthreads())

  return GLMBlock1D(link, distrib, phi, coef, vcov, iter, relErr, absErr, converged, 
             dev, residuals)
end

