#=
  Data Import to Julia
=#

using DelimitedFiles: readdlm, writedlm

path = "/home/chib/code/glmSolver/"

# Read the data

# Energy data
energyX = readdlm(path*"data/energyX.csv", ',', Float64);
energyY = readdlm(path*"data/energyY.csv", ',', Float64);

# Insurance data
insuranceX = readdlm(path*"data/insuranceX.csv", ',', Float64);
insuranceY = readdlm(path*"data/insuranceY.csv", ',', Float64);

# Credit Card Fraud
creditX = readdlm(path*"data/creditX.csv", ',', Float64);
creditY = readdlm(path*"data/creditY.csv", ',', Float64);

# GPA Data
gpaX = readdlm(path*"data/gpaX.csv", ',', Float64);
gpaY = readdlm(path*"data/gpaY.csv", ',', Float64);

# Cars Data
carsX = readdlm(path*"data/carsX.csv", ',', Float64);
carsY = readdlm(path*"data/carsY.csv", ',', Float64);

# Quine Data for negative Binomial Distribution
quineX = readdlm(path*"data/quineX.csv", ',', Float64);
quineY = readdlm(path*"data/quineY.csv", ',', Float64);

# Education data for Binomial Distribution
educationX = readdlm(path*"data/educationX.csv", ',', Float64);
educationY = readdlm(path*"data/educationY.csv", ',', Float64);

# Block data
creditBlockX = readBlockMatrix(Float64, path * "data/creditBlockX")
creditBlockY = readBlockMatrix(Float64, path * "data/creditBlockY")

educationBlockX = readBlockMatrix(Float64, path * "data/educationBlockX")
educationBlockY = readBlockMatrix(Float64, path * "data/educationBlockY")

energyBlockX = readBlockMatrix(Float64, path * "data/energyBlockX")
energyBlockY = readBlockMatrix(Float64, path * "data/energyBlockY")

insuranceBlockX = readBlockMatrix(Float64, path * "data/insuranceBlockX")
insuranceBlockY = readBlockMatrix(Float64, path * "data/insuranceBlockY")


