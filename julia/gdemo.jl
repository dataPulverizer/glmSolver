#=
  Demos for gradient descent 
=#
path = "/home/chib/code/glmSolver/"

include(path * "julia/GLMSolver.jl")
using .GLMSolver

using DelimitedFiles: readdlm, writedlm

function gdDataDemo()

  path = "/home/chib/code/glmSolver/data/";

  energyBlockX = readBlockMatrix(Float64, path * "energyScaledBlockX/");
  energyBlockY = readBlockMatrix(Float64, path * "energyScaledBlockY/");
  
  energyX = read2DArray(Float64, path * "energyScaledX.bin");
  energyY = read2DArray(Float64, path * "energyScaledY.bin");

  #= Number of parameters =#
  p = size(energyX)[2]

  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GESVSolver(), inverse = GETRIInverse())
  println("Full GLM Solve\n", gammaModel.coefficients)

  println("The outputs for all these models should be the same.");
  gammaModel = glm(RegularData(), energyX, 
        energyY, GammaDistribution(), LogLink(),
        #= solver =# GradientDescentSolver(1E-6), inverse = GETRIInverse(),
        control = Control{Float64}(maxit = 50), 
        calculateCovariance = true, 
        doStepControl = false)
  println("Gradient Descent With Regular Data\n", gammaModel.coefficients)
  #===================================================================#
  #= Gradient Descent Block Model =#
  gammaModel = glm(Block1D(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GradientDescentSolver(1E-6), 
        inverse = GETRIInverse(), control = Control{Float64}(maxit = 50), 
        calculateCovariance = true, 
        doStepControl = false)
  println("Gradient Descent With Block Data \n", gammaModel.coefficients)
  
  #= Gradient Descent Nesterov Block Model =#
  gammaModel = glm(Block1DParallel(), energyBlockX, 
        energyBlockY, GammaDistribution(), LogLink(),
        #= solver =# GradientDescentSolver(1E-6),
        inverse = GETRIInverse(), control = Control{Float64}(maxit = 50), 
        calculateCovariance = true, 
        doStepControl = false)
 println("Gradient Descent With Parallel Block Data \n", gammaModel.coefficients)
  
  return
end
