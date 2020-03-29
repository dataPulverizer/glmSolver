#=
  Demos for gradient descent 
=#
path = "/home/chib/code/glmSolver/"

include(path * "julia/GLMSolver.jl")
using .GLMSolver
using Random: seed!

seed!(0);
gammaLogX, gammaLogY = simulateData(Float64, GammaDistribution(), LogLink(), 30, 100_000);
gammaLogBlockX = matrixToBlock(gammaLogX, 100);
gammaLogBlockY = matrixToBlock(gammaLogY, 100);

function gdDataDemo(niter::Int64 = 20, learningRate::Float64 = 2E-8)
  
  #= Number of parameters =#
  p = size(gammaLogX)[2]
  
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)
  
  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), gammaLogX, 
        gammaLogY, GammaDistribution(), LogLink(),
        #= solver =# GradientDescentSolver(learningRate), inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# GradientDescentSolver(learningRate), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# GradientDescentSolver(learningRate),
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end

function momentumDataDemo(niter::Int64 = 20, learningRate::Float64 = 2E-8, momentum::Float64 = 0.7)
  
  #= Number of parameters =#
  p = size(gammaLogX)[2]
  
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)
  
  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), gammaLogX, 
      gammaLogY, GammaDistribution(), LogLink(),
      #= solver =# MomentumSolver(learningRate, momentum, p), 
      inverse = GETRIInverse(),
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# MomentumSolver(learningRate, momentum, p), 
      inverse = GETRIInverse(), 
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# MomentumSolver(learningRate, momentum, p),
      inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function nesterovDataDemo(niter::Int64 = 30, learningRate::Float64 = 2E-8, momentum::Float64 = 0.70)

  #= Number of parameters =#
  p = size(gammaLogX)[2]

  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), gammaLogX, 
      gammaLogY, GammaDistribution(), LogLink(),
      #= solver =# NesterovSolver(learningRate, momentum, p), 
      inverse = GETRIInverse(),
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# NesterovSolver(learningRate, momentum, p), 
      inverse = GETRIInverse(), 
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# NesterovSolver(learningRate, momentum, p),
      inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function adagradDataDemo(niter::Int64 = 10, learningRate::Float64 = 1E-10, epsilon::Float64 = 1E-8)
  
  #= Number of parameters =#
  p = size(gammaLogX)[2]
  
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)
  
  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), gammaLogX, 
      gammaLogY, GammaDistribution(), LogLink(),
      #= solver =# AdagradSolver(learningRate, p, epsilon), 
      inverse = GETRIInverse(),
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# AdagradSolver(learningRate, p, epsilon), 
      inverse = GETRIInverse(), 
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# AdagradSolver(learningRate, p, epsilon),
      inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end

#= Poor convergence performance for this one =#
function adadeltaDataDemo(niter::Int64 = 10, momentum::Float64 = 0.9, epsilon::Float64 = 1E-8)
  
  #= Number of parameters =#
  p = size(gammaLogX)[2]
  
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)
  
  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), gammaLogX, 
      gammaLogY, GammaDistribution(), LogLink(),
      #= solver =# AdadeltaSolver(momentum, p, epsilon), 
      inverse = GETRIInverse(),
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# AdadeltaSolver(momentum, p, epsilon), 
      inverse = GETRIInverse(), 
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
      gammaLogBlockY, GammaDistribution(), LogLink(),
      #= solver =# AdadeltaSolver(momentum, p, epsilon),
      inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function RMSPropDataDemo(niter::Int64 = 10, learningRate::Float64 = 1E-2, epsilon::Float64 = 1E-8)
  
  #= Number of parameters =#
  p = size(gammaLogX)[2]
  
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)
  
  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), gammaLogX, 
        gammaLogY, GammaDistribution(), LogLink(),
        #= solver =# RMSpropSolver(learningRate, p, epsilon), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# RMSpropSolver(learningRate, p, epsilon), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# RMSpropSolver(learningRate, p, epsilon),
        inverse = GETRIInverse(), 
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function adamDataDemo(niter::Int64 = 50, learningRate::Float64 = 1E-3, b1::Float64 = 0.9, b2::Float64 = 0.999, epsilon::Float64 = 1E-8)
  
  #= Number of parameters =#
  p = size(gammaLogX)[2]
  
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)
  
  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), gammaLogX, 
        gammaLogY, GammaDistribution(), LogLink(),
        #= solver =# AdamSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# AdamSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# AdamSolver(learningRate, b1, b2, p, epsilon),
        inverse = GETRIInverse(), 
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function adaMaxDataDemo(niter::Int64 = 10, learningRate::Float64 = 1E-1, b1::Float64 = 0.9, b2::Float64 = 0.999)
  
  #= Number of parameters =#
  p = size(gammaLogX)[2]
  
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)
  
  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), gammaLogX, 
        gammaLogY, GammaDistribution(), LogLink(),
        #= solver =# AdaMaxSolver(learningRate, b1, b2, p), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# AdaMaxSolver(learningRate, b1, b2, p), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# AdaMaxSolver(learningRate, b1, b2, p),
        inverse = GETRIInverse(), 
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function nadamDataDemo(niter::Int64 = 30, learningRate::Float64 = 3E-3, b1::Float64 = 0.9, b2::Float64 = 0.999, epsilon::Float64 = 1E-8)
  
  #= Number of parameters =#
  p = size(gammaLogX)[2]
  
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)
  
  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), gammaLogX, 
        gammaLogY, GammaDistribution(), LogLink(),
        #= solver =# NAdamSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# NAdamSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# NAdamSolver(learningRate, b1, b2, p, epsilon),
        inverse = GETRIInverse(), 
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end

function amsGradDataDemo(niter::Int64 = 50, learningRate::Float64 = 9E-3, b1::Float64 = 0.9, b2::Float64 = 0.999, epsilon::Float64 = 1E-8)
  
  #= Number of parameters =#
  p = size(gammaLogX)[2]
  
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)
  
  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), gammaLogX, 
        gammaLogY, GammaDistribution(), LogLink(),
        #= solver =# AMSGradSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# AMSGradSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), gammaLogBlockX, 
        gammaLogBlockY, GammaDistribution(), LogLink(),
        #= solver =# AMSGradSolver(learningRate, b1, b2, p, epsilon),
        inverse = GETRIInverse(), 
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end

#===================================================================#

# Simple Gradient Descent
gdDataDemo(30, 2E-8)
# Gradient Descent with Momentum
momentumDataDemo(20, 2E-8, 0.7)
# Nesterov
nesterovDataDemo(30, 2E-8, 0.7)
# Adagrad
adagradDataDemo(10, 1E-10, 1E-8)
# Adadelta (Poor performance)
adadeltaDataDemo(10, 0.9, 1E-8)
# RMSProp
RMSPropDataDemo(10, 1E-2, 1E-8)
# Adam
adamDataDemo(50, 1E-3, 0.9, 0.999, 1E-8)
# AdaMax
adaMaxDataDemo(10, 1E-1, 0.9, 0.999)
# NAdam
nadamDataDemo(30, 3E-3, 0.9, 0.999, 1E-8)
# AMSGrad
amsGradDataDemo(40, 5E-4, 0.9, 0.999, 1E-8)


