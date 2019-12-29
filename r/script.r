# Data sources:
#1. US Energy data
#

# Bug fixes
# Complementary log-log - work in progress
# binomialCloglogR = matrix(glm.fit(binomialX, binomialY, family = binomial("cloglog"))$coefficients)
# binomialCloglogCpp = protoSimpleGLM(binomialX, binomialY, "cloglog", "Binomial", FALSE)

# This is an example of where probit fails
# xR = matrix(glm.fit(binomialX, binomialY, family = binomial("probit"))$coefficients)
# xCpp = protoSimpleGLM(binomialX, binomialY, "probit", "Binomial", TRUE)


require(data.table)

setGeneric("%+%", def = function(e1, e2) standardGeneric("%+%"))
setMethod("%+%", signature = c("character", "character"), definition = function(e1, e2) paste0(e1, e2))

path = "/home/chib/code/GLMPrototype/"
Rcpp::sourceCpp(path %+% "cpp/glm.cpp")

# Error functions
absoluteError = function(x1, x2) sqrt(sum((x1 - x2)^2))
relativeError = function(x1, x2) absoluteError(x1, x2)/norm(x1)

# Load the datasets

# Read energy data for testing
energyData = fread(path %+% "data/modelFrame.csv")
# Format the data
energyData[, Year := factor(Year)]
energyData[, Day := factor(Day)]
energyData[, HourFactor := factor(HourFactor)]
energyData[, MonthName := factor(MonthName)]

# Credit fraud data
cfData = fread(path %+% "data/creditfraudModelFrame.csv")
# GPA data
gpaData = fread(path %+% "data/gpaData.csv")
gpaData[, rank := factor(rank)]

# Cars Data
Cars = data.table(mtcars)
Cars[, cyl := factor(cyl)]
Cars[, vs := factor(vs)]
Cars[, am := factor(am)]
Cars[, gear := factor(gear)]
Cars[, carb := factor(carb)]


# Create the model matrices

# For Gamma family with log link
modelFormula1 = formula(paste0("~ ", paste0(c("Year", "Day", "HourFactor", 
                       "MonthName", paste0("EnergyLag", 24:47)), collapse = " + ")))
energyX = model.matrix(modelFormula1, data = data.frame(energyData))
energyY = model.matrix(~ EnergyMegaWattHours - 1, data = data.frame(energyData))

# Basic tests for Poisson Model
poissonX = model.matrix(~ Kilometres + Zone + Bonus + Make + Insured, data = faraway::motorins)
poissonY = model.matrix(~ Claims - 1, data = faraway::motorins)

# Basic test for Binomial Model
binomialX = model.matrix(Class ~ ., data = cfData)
binomialY = model.matrix(~ Class - 1, data = cfData)

# GPA Data Model Matrices
gpaX = model.matrix(admit ~ ., data = gpaData)
gpaY = model.matrix(~ admit - 1, data = gpaData)

# Cars data
carsX = model.matrix(mpg ~ ., data = mtcars)
carsY = model.matrix(~ mpg - 1, data = mtcars)

# Credit fraud data
cfData = fread(paste0(dataFolder, "creditfraudModelFrame.csv"))
# Basic test for Binomial Model
creditX = model.matrix(Class ~ ., data = cfData)
creditY = model.matrix(~ Class - 1, data = cfData)


# The GLM Models

# Gamma GLM

# Log link
# Native R GLM
betaR = matrix(glm.fit(energyX, energyY, family = Gamma("log"))$coefficients)
# The C++ version
betaCpp = protoSimpleGLM(energyX, energyY, "log", "Gamma", FALSE)

# Discrepancy between the different coefficients
print(paste0("Absolute Error: ", signif(absoluteError(betaR, betaCpp), 3)))
print(paste0("Relative Error: ", signif(relativeError(betaR, betaCpp), 3)))

# Inverse link
# Native R GLenergyX
print(paste0("Absolute Error: ", signif(absoluteError(gammaInverseR, gammaInverseCpp), 3)))
print(paste0("Relative Error: ", signif(relativeError(gammaInverseR, gammaInverseCpp), 3)))


# Poisson Regression
# Native R GLM
poissonLogR = matrix(glm.fit(poissonX, poissonY, family = poisson("log"))$coefficients)
# The C++ version
poissonLogCpp = protoSimpleGLM(poissonX, poissonY, "log", "Poisson", FALSE)
# Discrepancy
print(paste0("Absolute Error: ", signif(absoluteError(poissonLogR, poissonLogCpp), 3)))
print(paste0("Relative Error: ", signif(relativeError(poissonLogR, poissonLogCpp), 3)))

# Binomial Regression

# Logit link
# R GLM
binomialLogitR = matrix(glm.fit(binomialX, binomialY, family = binomial("logit"))$coefficients)
# The C++ version
binomialLogitCpp = protoSimpleGLM(binomialX, binomialY, "logit", "Binomial", FALSE)
# Discrepancy
print(paste0("Absolute Error: ", signif(absoluteError(poissonLogR, poissonLogCpp), 3)))
print(paste0("Relative Error: ", signif(relativeError(poissonLogR, poissonLogCpp), 3)))

# Probit Link
# R Version
binomialProbitR = matrix(glm.fit(gpaX, gpaY, family = binomial("probit"))$coefficients)
# The C++ version
binomialProbitCpp = protoSimpleGLM(gpaX, gpaY, "probit", "Binomial", FALSE)
# Discrepancy
print(paste0("Absolute Error: ", signif(absoluteError(binomialProbitR, binomialProbitCpp), 3)))
print(paste0("Relative Error: ", signif(relativeError(binomialProbitR, binomialProbitCpp), 3)))


# Power GLM - doesn't return the right coefficients in R!
gammaPowerM2R = matrix(glm.fit(energyX, energyY, family = Gamma(power(-2)))$coefficients)

# Negative binomial
require(MASS)
quineX = model.matrix(~ ., data = quine[quine$Days > 0,])
quineY = model.matrix(~ Days - 1, data = quine[quine$Days > 0,])
negativeBinomialR = matrix(glm.fit(quineX, quineY, family = negative.binomial(2))$coefficients)
negativeBinomialRCars = matrix(glm.fit(carsX, carsY, family = negative.binomial(1.0))$coefficients)



# Cauchit Link with binomial distribution
education = fread(paste0(path, "data/education.csv"), sep = ",")
educationX = model.matrix( ~ age + education + wantsMore, data = education)
educationY = model.matrix(~ cbind(using, notUsing) - 1, data = education)

# Just to see whether our implementation can handle 
#    two columns for a binomial distribution
binomialCountTest = glm.fit(educationX, educationY, family = binomial("logit"), control = list(trace = TRUE))$coefficients

cauchitR = glm.fit(educationX, educationY, family = binomial("cauchit"), control = list(trace = TRUE))$coefficients
binomial_probit_link = glm.fit(gpaX, gpaY, family = binomial("probit"))
gaussian_log_link = glm.fit(energyX, energyY, family = gaussian("log"))



