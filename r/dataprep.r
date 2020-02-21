#= Purpose is to prepare the test data in for use in Julia =#
require(MASS)
require(data.table)

source("/home/chib/code/glmSolver/r/paths.r")
source(rCodeFolder %+% "binaryIO.r")

# Read and prep data then write out

# Energy data
energyData = fread(paste0(dataFolder, "modelFrame.csv"))
# Format the data
energyData[, Year := factor(Year)]
energyData[, Day := factor(Day)]
energyData[, HourFactor := factor(HourFactor)]
energyData[, MonthName := factor(MonthName)]

modelFormula1 = formula(paste0("~ ", paste0(c("Year", "Day", "HourFactor", 
                       "MonthName", paste0("EnergyLag", 24:47)), collapse = " + ")))
energyX = model.matrix(modelFormula1, data = data.frame(energyData))
energyY = model.matrix(~ EnergyMegaWattHours - 1, data = data.frame(energyData))

# Write as text files
fwrite(data.table(energyX), file = paste0(dataFolder, "energyX.csv"), sep = ",", col.names = FALSE)
fwrite(data.table(energyY), file = paste0(dataFolder, "energyY.csv"), sep = ",", col.names = FALSE)
# Write as binary files
write2DArray(dataFolder %+% "energyX.bin", energyX, 8)
write2DArray(dataFolder %+% "energyY.bin", energyY, 8)
# write1DArray(dataFolder %+% "energyY.bin", as.vector(energyY), 8)

# Write Matrix as block
write2DBlock(dataFolder %+% "energyBlockX", matrixToBlock(energyX, 1000))
write2DBlock(dataFolder %+% "energyBlockY", matrixToBlock(energyY, 1000))

# Scaled X Matrix
scaledMatrix = cbind(energyX[,1], scale(energyX[,-1]))
write2DArray(dataFolder %+% "energyScaledX.bin", scaledMatrix, 8)
write2DBlock(dataFolder %+% "energyScaledBlockX", matrixToBlock(scaledMatrix, 1000))

# Motor insurance data
insuranceX = model.matrix(~ Kilometres + Zone + Bonus + Make + Insured, data = faraway::motorins)
insuranceY = model.matrix(~ Claims - 1, data = faraway::motorins)

# Write to file
fwrite(data.table(insuranceX), file = paste0(dataFolder, "insuranceX.csv"), sep = ",", col.names = FALSE)
fwrite(data.table(insuranceY), file = paste0(dataFolder, "insuranceY.csv"), sep = ",", col.names = FALSE)
# Write as binary files
write2DArray(dataFolder %+% "insuranceX.bin", insuranceX, 8)
write2DArray(dataFolder %+% "insuranceY.bin", insuranceY, 8)
# write1DArray(dataFolder %+% "insuranceY.bin", as.vector(insuranceY), 8)

# Write Matrix as block
write2DBlock(dataFolder %+% "insuranceBlockX", matrixToBlock(insuranceX, 20))
write2DBlock(dataFolder %+% "insuranceBlockY", matrixToBlock(insuranceY, 20))

# Scaled X Matrix
scaledMatrix = cbind(insuranceX[,1], scale(insuranceX[,-1]))
write2DArray(dataFolder %+% "insuranceScaledX.bin", scaledMatrix, 8)
write2DBlock(dataFolder %+% "insuranceScaledBlockX", matrixToBlock(scaledMatrix, 20))


# Credit fraud data
cfData = fread(paste0(dataFolder, "creditfraudModelFrame.csv"))
# Basic test for Binomial Model
creditX = model.matrix(Class ~ ., data = cfData)
creditY = model.matrix(~ Class - 1, data = cfData)

# Write to file
fwrite(data.table(creditX), file = paste0(dataFolder, "creditX.csv"), sep = ",", col.names = FALSE)
fwrite(data.table(creditY), file = paste0(dataFolder, "creditY.csv"), sep = ",", col.names = FALSE)
# fwrite(data.table(creditY), file = paste0(dataFolder, "creditY.csv"), sep = ",", col.names = FALSE)

# Write as binary files
write2DArray(dataFolder %+% "creditX.bin", creditX, 8)
write2DArray(dataFolder %+% "creditY.bin", creditY, 8)
# write1DArray(dataFolder %+% "creditY.bin", as.vector(creditY), 8)

# Write Matrix as block
write2DBlock(dataFolder %+% "creditBlockX", matrixToBlock(creditX, 200))
write2DBlock(dataFolder %+% "creditBlockY", matrixToBlock(creditY, 200))

# Scaled X Matrix
scaledMatrix = cbind(creditX[,1], scale(creditX[,-1]))
write2DArray(dataFolder %+% "creditScaledX.bin", scaledMatrix, 8)
write2DBlock(dataFolder %+% "creditScaledBlockX", matrixToBlock(scaledMatrix, 200))

# GPA data
gpaData = fread(paste0(dataFolder, "gpaData.csv"))
gpaData[, rank := factor(rank)]

# GPA Data Model Matrices
gpaX = model.matrix(admit ~ ., data = gpaData)
gpaY = model.matrix(~ admit - 1, data = gpaData)

# Write to file
fwrite(data.table(gpaX), file = paste0(dataFolder, "gpaX.csv"), sep = ",", col.names = FALSE)
fwrite(data.table(gpaY), file = paste0(dataFolder, "gpaY.csv"), sep = ",", col.names = FALSE)
# Write as binary files
write2DArray(dataFolder %+% "gpaX.bin", gpaX, 8)
write2DArray(dataFolder %+% "gpaY.bin", gpaY, 8)
# write1DArray(dataFolder %+% "gpaY.bin", as.vector(gpaY), 8)

# Scaled X Matrix
scaledMatrix = cbind(gpaX[,1], scale(gpaX[,-1]))
write2DArray(dataFolder %+% "gpaScaledX.bin", scaledMatrix, 8)

# Write the cars data to file
Cars = data.table(mtcars)
Cars[, cyl := factor(cyl)]
Cars[, vs := factor(vs)]
Cars[, am := factor(am)]
Cars[, gear := factor(gear)]
Cars[, carb := factor(carb)]
carsX = model.matrix(mpg ~ ., data = mtcars)
carsY = model.matrix(~ mpg - 1, data = mtcars)

# Write data to file
fwrite(data.table(carsX), file = paste0(dataFolder, "carsX.csv"), col.names = FALSE)
fwrite(data.table(carsY), file = paste0(dataFolder, "carsY.csv"), col.names = FALSE)
# Write as binary files
write2DArray(dataFolder %+% "carsX.bin", carsX, 8)
write2DArray(dataFolder %+% "carsY.bin", carsY, 8)
# write1DArray(dataFolder %+% "carsY.bin", as.vector(carsY), 8)

# Scaled X Matrix
scaledMatrix = cbind(carsX[,1], scale(carsX[,-1]))
write2DArray(dataFolder %+% "carsScaledX.bin", scaledMatrix, 8)
write2DBlock(dataFolder %+% "carsBlockY", matrixToBlock(carsY, 5))
write2DBlock(dataFolder %+% "carsScaledBlockX", matrixToBlock(scaledMatrix, 5))


# For negative binomial distribution
quineX = model.matrix(Days ~ ., data = quine[quine$Days > 0,])
quineY = model.matrix(~ Days - 1, data = quine[quine$Days > 0,])

# Write data to file
fwrite(data.table(quineX), file = paste0(dataFolder, "quineX.csv"), col.names = FALSE)
fwrite(data.table(quineY), file = paste0(dataFolder, "quineY.csv"), col.names = FALSE)
# Write as binary files
write2DArray(dataFolder %+% "quineX.bin", quineX, 8)
write2DArray(dataFolder %+% "quineY.bin", quineY, 8)
# write1DArray(dataFolder %+% "quineY.bin", as.vector(quineY), 8)

# Scaled X Matrix
write2DArray(dataFolder %+% "quineScaledX.bin", cbind(quineX[,1], scale(quineX[,-1])), 8)

# For binomial model
education = fread(paste0(dataFolder, "education.csv"), sep = ",")
educationX = model.matrix( ~ age + education + wantsMore, data = education)
educationY = model.matrix(~ cbind(using, notUsing) - 1, data = education)

# Write data to file
fwrite(data.table(educationX), file = paste0(dataFolder, "educationX.csv"), col.names = FALSE)
fwrite(data.table(educationY), file = paste0(dataFolder, "educationY.csv"), col.names = FALSE)
# Write as binary files
write2DArray(dataFolder %+% "educationX.bin", educationX, 8)
write2DArray(dataFolder %+% "educationY.bin", educationY, 8)
# write1DArray(dataFolder %+% "educationY.bin", as.vector(educationY), 8)

# Write Matrix as block
write2DBlock(dataFolder %+% "educationBlockX", matrixToBlock(educationX, 8))
write2DBlock(dataFolder %+% "educationBlockY", matrixToBlock(educationY, 8))

# Scaled X Matrix
scaledMatrix = cbind(educationX[,1], scale(educationX[,-1]))
write2DArray(dataFolder %+% "educationScaledX.bin", scaledMatrix, 8)
write2DBlock(dataFolder %+% "educationScaledBlockX", matrixToBlock(scaledMatrix, 8))

