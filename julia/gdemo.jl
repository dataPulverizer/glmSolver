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

