#=
  Demos for gradient descent 
=#
path = "/home/chib/code/glmSolver/"

include(path * "julia/GLMSolver.jl")
using .GLMSolver
using DelimitedFiles: readdlm, writedlm

function gdDataDemo(niter::Int64 = 10, learningRate::Float64 = 3E-6)

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), energyX, 
        energyY, GammaDistribution(), LogLink(),
        #= solver =# GradientDescentSolver(learningRate), inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GradientDescentSolver(learningRate), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GradientDescentSolver(learningRate),
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end

function momentumDataDemo(niter::Int64 = 10, learningRate::Float64 = 3E-7, momentum::Float64 = 0.7)

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), energyX, 
      energyY, GammaDistribution(), LogLink(),
      #= solver =# MomentumSolver(learningRate, momentum, p), 
      inverse = GETRIInverse(),
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# MomentumSolver(learningRate, momentum, p), 
      inverse = GETRIInverse(), 
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# MomentumSolver(learningRate, momentum, p),
      inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function nesterovDataDemo(niter::Int64 = 10, learningRate::Float64 = 3E-7, momentum::Float64 = 0.75)

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), energyX, 
      energyY, GammaDistribution(), LogLink(),
      #= solver =# NesterovSolver(learningRate, momentum, p), 
      inverse = GETRIInverse(),
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# NesterovSolver(learningRate, momentum, p), 
      inverse = GETRIInverse(), 
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# NesterovSolver(learningRate, momentum, p),
      inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function adagradDataDemo(niter::Int64 = 10, learningRate::Float64 = 1E-9, epsilon::Float64 = 1E-8)

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), energyX, 
      energyY, GammaDistribution(), LogLink(),
      #= solver =# AdagradSolver(learningRate, p, epsilon), 
      inverse = GETRIInverse(),
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# AdagradSolver(learningRate, p, epsilon), 
      inverse = GETRIInverse(), 
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# AdagradSolver(learningRate, p, epsilon),
      inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end

#= This one seems kind of rubbish =#
function adadeltaDataDemo(niter::Int64 = 10, momentum::Float64 = 0.9, epsilon::Float64 = 1E-8)

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), energyX, 
      energyY, GammaDistribution(), LogLink(),
      #= solver =# AdadeltaSolver(momentum, p, epsilon), 
      inverse = GETRIInverse(),
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# AdadeltaSolver(momentum, p, epsilon), 
      inverse = GETRIInverse(), 
      control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
      energyBlockY, GammaDistribution(), LogLink(),
      #= solver =# AdadeltaSolver(momentum, p, epsilon),
      inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
      calculateCovariance = true, 
      doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function RMSPropDataDemo(niter::Int64 = 50, learningRate::Float64 = 1.3E-2, epsilon::Float64 = 1E-8)

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), energyX, 
        energyY, GammaDistribution(), LogLink(),
        #= solver =# RMSpropSolver(learningRate, p, epsilon), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# RMSpropSolver(learningRate, p, epsilon), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# RMSpropSolver(learningRate, p, epsilon),
        inverse = GETRIInverse(), 
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function adamDataDemo(niter::Int64 = 50, learningRate::Float64 = 3E-2, b1::Float64 = 0.9, b2::Float64 = 0.999, epsilon::Float64 = 1E-8)

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), energyX, 
        energyY, GammaDistribution(), LogLink(),
        #= solver =# AdamSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# AdamSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# AdamSolver(learningRate, b1, b2, p, epsilon),
        inverse = GETRIInverse(), 
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function adaMaxDataDemo(niter::Int64 = 10, learningRate::Float64 = 1E-1, b1::Float64 = 0.9, b2::Float64 = 0.999)

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), energyX, 
        energyY, GammaDistribution(), LogLink(),
        #= solver =# AdaMaxSolver(learningRate, b1, b2, p), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# AdaMaxSolver(learningRate, b1, b2, p), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# AdaMaxSolver(learningRate, b1, b2, p),
        inverse = GETRIInverse(), 
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end


function nadamDataDemo(niter::Int64 = 50, learningRate::Float64 = 3E-2, b1::Float64 = 0.9, b2::Float64 = 0.999, epsilon::Float64 = 1E-8)

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), energyX, 
        energyY, GammaDistribution(), LogLink(),
        #= solver =# NAdamSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# NAdamSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# NAdamSolver(learningRate, b1, b2, p, epsilon),
        inverse = GETRIInverse(), 
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end

function amsGradDataDemo(niter::Int64 = 50, learningRate::Float64 = 9E-3, b1::Float64 = 0.9, b2::Float64 = 0.999, epsilon::Float64 = 1E-8)

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse());
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  
  gammaModel = glm(RegularData(), energyX, 
        energyY, GammaDistribution(), LogLink(),
        #= solver =# AMSGradSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(),
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# AMSGradSolver(learningRate, b1, b2, p, epsilon), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Block1DParallel Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# AMSGradSolver(learningRate, b1, b2, p, epsilon),
        inverse = GETRIInverse(), 
        control = Control{Float64}(maxit = niter), 
        calculateCovariance = true, 
        doStepControl = false);
  println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end

amsGradDataDemo(50, 3E-2, 0.9, 0.999, 1E-8)

#===================================================================#

using Random: seed!
seed!(0);
gammaLogX, gammaLogY = simulateData(Float64, GammaDistribution(), LogLink(), 30, 100_000);
gammaLogBlockX = matrixToBlock(gammaLogX, 100);
gammaLogBlockY = matrixToBlock(gammaLogY, 100);

function gdDataDemo2(niter::Int64 = 50, learningRate::Float64 = 1E-8)

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
