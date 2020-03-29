module demos.demo;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;
import glmsolverd.link;
import glmsolverd.distributions;
import glmsolverd.tools;
import glmsolverd.linearalgebra;
import glmsolverd.io;
import glmsolverd.fit;
import glmsolverd.sample;

import std.conv: to;
import std.stdio : writeln;
import std.file: remove;
import std.parallelism;

/* ldc2 demos.d arrays.d arraycommon.d apply.d link.d distributions.d tools.d linearalgebra.d io.d fit.d -O2 -L-lopenblas -L-lpthread -L-llapacke -L-llapack -L-lm && ./demos */

/* Timed demo for basic observational benchmarking */
void timed_demo()
{ 
  /* Generate the data */
  ulong seed = 3;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double, CblasColMajor)(distrib, link, 60, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  
  //string path = "/home/chib/code/glmSolver/data/";
  //writeMatrix(path ~ "gammaX.bin", gammaX);
  //writeMatrix(path ~ "gammaY.bin", gammaY);
  
  /* Gamma Distribution With Log Link */
  import std.datetime.stopwatch : AutoStart, StopWatch;
  openblas_set_num_threads(1); /* Set the number of cores used to 1 */
  auto sw = StopWatch(AutoStart.no);
  sw.start();
  auto gamma_distrib_log_link = glm(
      new RegularData(), gammaX, gammaY, 
      distrib, link, new GESVSolver!(double)(), 
      new GETRIInverse!(double)());
  sw.stop();
  writeln(gamma_distrib_log_link);
  writeln("Time taken: ", sw.peek.total!"msecs", " msec");
  
  return;
}

void parallelBlockGLMDemo()
{
  /* Simulate the data */
  auto seed = 3;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);
  
  seed = 4;
  distrib = new BinomialDistribution!(double)();
  link = new LogitLink!(double)();
  auto binomialData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto binomialX = binomialData.X;
  auto binomialY = binomialData.y;
  auto binomialBlockX = matrixToBlock(binomialX, 10);
  auto binomialBlockY = matrixToBlock(binomialY, 10);
  
  seed = 5;
  distrib = new GaussianDistribution!(double)();
  link = new IdentityLink!(double)();
  auto gaussianData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gaussianX = gaussianData.X;
  auto gaussianY = gaussianData.y;
  auto gaussianBlockX = matrixToBlock(gaussianX, 10);
  auto gaussianBlockY = matrixToBlock(gaussianY, 10);
  
  seed = 6;
  distrib = new PoissonDistribution!(double)();
  link = new LogLink!(double)();
  auto poissonData = simulateData!(double)(distrib, link, 5, 100, seed);
  auto poissonX = poissonData.X;
  auto poissonY = poissonData.y;
  auto poissonBlockX = matrixToBlock(poissonX, 10);
  auto poissonBlockY = matrixToBlock(poissonY, 10);
  /***************************************************************/
  /* Gamma Model With Log Link */
  auto glmModel = glm(new RegularData(), gammaX, 
        gammaY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  
  auto blockGLMModel = glm(new Block1D(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  
  auto blockParallelGLMModel = glm(new Block1DParallel(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  /***************************************************************/
  /* Gaussian Model With Identity Link */

  glmModel = glm!(double)(new RegularData(), gaussianX, 
        gaussianY, new GaussianDistribution!(double)(), new IdentityLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  
  blockGLMModel = glm!(double)(new Block1D(), gaussianBlockX, 
        gaussianBlockY, new GaussianDistribution!(double)(), new IdentityLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  
  blockParallelGLMModel = glm!(double)(new Block1DParallel(), gaussianBlockX, 
        gaussianBlockY, new GaussianDistribution!(double)(), new IdentityLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  /***************************************************************/
  /* Binomial Model With Logit Link */
  glmModel = glm!(double)(new RegularData(), binomialX, 
        binomialY, new BinomialDistribution!(double)(), new LogitLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  
  blockGLMModel = glm!(double)(new Block1D(), binomialBlockX, 
        binomialBlockY, new BinomialDistribution!(double)(), new LogitLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  
  blockParallelGLMModel = glm!(double)(new Block1DParallel(), binomialBlockX, 
        binomialBlockY, new BinomialDistribution!(double)(), new LogitLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  /***************************************************************/
  /* Poisson Model With Log Link */

  glmModel = glm!(double)(new RegularData(), poissonX, 
        poissonY, new PoissonDistribution!(double)(), new LogLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  
  blockGLMModel = glm!(double)(new Block1D(), poissonBlockX, 
        poissonBlockY, new PoissonDistribution!(double)(), new LogLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  
  blockParallelGLMModel = glm!(double)(new Block1DParallel(), poissonBlockX, 
        poissonBlockY, new PoissonDistribution!(double)(), new LogLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  /***************************************************************/
}

/* Gradient Descent for all data types */
void gdDataDemo()
{
  /* Simulate the data */
  //writeln("Generating Gamma Data");
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  //writeln("Converting matrix to block matrices");
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);

  /* Number of parameters */
  auto p = gammaX.ncol;
  /***************************************************************/
  /* Gradient Descent With Regular Data */
  auto gammaModel = glm!(double)(new RegularData(), gammaX, 
        gammaY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  writeln("Regular Model\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent With Regular Data */
  gammaModel = glm!(double)(new RegularData(), gammaX, 
        gammaY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GradientDescent!(double)(1E-4), new GETRIInverse!(double)(),
        new Control!(double)(30));
  writeln("Gradient Descent solver with regular data \n", gammaModel);
  /***************************************************************/
  /* Gradient Descent With Block Data */
  gammaModel = glm!(double)(new Block1D(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GradientDescent!(double)(1E-4), new GETRIInverse!(double)(),
        new Control!(double)(30));
  writeln("Gradient Descent solver with block data \n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Parallel Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GradientDescent!(double)(1E-4), new GETRIInverse!(double)(),
        new Control!(double)(30));
  writeln("Gradient Descent solver with block parallel data \n", gammaModel);
}

void gdMomentumDemo()
{
  /* Simulate the data */
  //writeln("Generating Gamma Data");
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  //writeln("Converting matrix to block matrices");
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);

  /* Number of parameters */
  auto p = gammaX.ncol;
  
  /***************************************************************/
  /* Standard GLM Model Using Regular Data */
  auto gammaModel = glm!(double)(new RegularData(), gammaX, 
        gammaY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  writeln("Regular Model\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GradientDescent!(double)(1E-4), new GETRIInverse!(double)(),
        new Control!(double)(30));
  writeln("Gradient Descent solver with parallel data \n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Momentum Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Momentum!(double)(2E-5, 0.90, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30));
  writeln("Gradient Descent solver with parallel data for Momentum Solver \n", gammaModel);
}

/* Test all the data types for momentum */
void gdMomentumDataDemo()
{
  /* Simulate the data */
  //writeln("Generating Gamma Data");
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  //writeln("Converting matrix to block matrices");
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);

  /* Number of parameters */
  auto p = gammaX.ncol;

  writeln("The outputs for all these models should be the same.");
  auto gammaModel = glm!(double)(new RegularData(), gammaX, 
        gammaY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Momentum!(double)(2E-5, 0.9, p), new GETRIInverse!(double)(),
        new Control!(double)(30));
  writeln("Momentum Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Momentum!(double)(2E-5, 0.9, p), new GETRIInverse!(double)(),
        new Control!(double)(30));
  writeln("Momentum Gradient Descent With Parallel Data \n", gammaModel);

  /* Gradient Descent Momentum Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Momentum!(double)(2E-5, 0.9, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30));
  writeln("Momentum Gradient Descent With Parallel Data \n", gammaModel);
}

/* Test all the data types for Nesterov */
void gdNesterovDataDemo()
{
  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);
  
  /* Number of parameters */
  auto p = gammaX.ncol;
  
  auto gammaModel = glm!(double)(new Block1DParallel(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GESVSolver!(double)(), new GETRIInverse!(double)());
  writeln("Full GLM Solve\n", gammaModel);

  writeln("The outputs for all these models should be the same.");
  gammaModel = glm!(double)(new RegularData(), gammaX, 
        gammaY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.3, p), new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("Nesterov Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(new Block1D(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.3, p), new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("Nesterov Gradient Descent With Block Data \n", gammaModel);
  
  /* Gradient Descent Nesterov Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), gammaBlockX, 
        gammaBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.3, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30), 
        totalCPUs, true, false);
  writeln("Nesterov Gradient Descent With Parallel Block Data \n", gammaModel);
}

/* Demo all the data types for Adagrad */
void gdAdagradDataDemo()
{
  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);
  
  /* Number of parameters */
  auto p = gammaX.ncol;
  
  auto gammaModel = glm!(double)(
        new Block1DParallel(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new GESVSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Full GLM Solve\n", gammaModel);

  writeln("The outputs for all these models should be the same.");
  gammaModel = glm!(double)(
        new RegularData(),
        gammaX, gammaY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new Adagrad!(double)(2E-9, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("Adagrad Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(
        new Block1D(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new Adagrad!(double)(2E-9, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("Adagrad Gradient Descent With Block Data \n", gammaModel);
  
  /* Gradient Descent Adagrad Block Model */
  gammaModel = glm!(double)(
        new Block1DParallel(), 
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new Adagrad!(double)(2E-9, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30), 
        totalCPUs, true, false);
  writeln("Adagrad Gradient Descent With Parallel Block Data \n", gammaModel);
}

/* Demo all the data types for Adadelta */
void gdAdadeltaDataDemo()
{
  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);
  
  /* Number of parameters */
  auto p = gammaX.ncol;
  
  auto gammaModel = glm!(double)(
        new Block1DParallel(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new GESVSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Full GLM Solve\n", gammaModel);

  writeln("The outputs for all these models should be the same.");
  gammaModel = glm!(double)(
        new RegularData(),
        gammaX, gammaY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new Adadelta!(double)(0.86, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("Adadelta Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(
        new Block1D(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new Adadelta!(double)(0.86, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("Adadelta Gradient Descent With Block Data \n", gammaModel);
  
  /* Gradient Descent Adadelta Block Model */
  gammaModel = glm!(double)(
        new Block1DParallel(), 
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new Adadelta!(double)(0.86, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30), 
        totalCPUs, true, false);
  writeln("Adadelta Gradient Descent With Parallel Block Data \n", gammaModel);
}

/* Demo all the data types for RMSProp */
void gdRMSPropDataDemo()
{
  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);
  
  /* Number of parameters */
  auto p = gammaX.ncol;
  
  auto gammaModel = glm!(double)(
        new Block1DParallel(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new GESVSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Full GLM Solve\n", gammaModel);

  writeln("The outputs for all these models should be the same.");
  gammaModel = glm!(double)(
        new RegularData(),
        gammaX, gammaY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new RMSprop!(double)(0.1, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("RMSprop Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(
        new Block1D(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new RMSprop!(double)(0.1, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("RMSprop Gradient Descent With Block Data \n", gammaModel);
  
  /* Gradient Descent RMSprop Block Model */
  gammaModel = glm!(double)(
        new Block1DParallel(), 
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new RMSprop!(double)(0.1, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30), 
        totalCPUs, true, false);
  writeln("RMSprop Gradient Descent With Parallel Block Data \n", gammaModel);
}

/* Demo all the data types for Adam */
void gdAdamDataDemo()
{
  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);
  
  /* Number of parameters */
  auto p = gammaX.ncol;
  
  auto gammaModel = glm!(double)(
        new Block1DParallel(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new GESVSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Full GLM Solve\n", gammaModel);

  writeln("The outputs for all these models should be the same.");
  gammaModel = glm!(double)(
        new RegularData(),
        gammaX, gammaY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new Adam!(double)(1E-6, 0.9, 0.999, 1E-6, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("Adam Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(
        new Block1D(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new Adam!(double)(1E-6, 0.9, 0.999, 1E-6, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("Adam Gradient Descent With Block Data \n", gammaModel);
  
  /* Gradient Descent Adam Block Model */
  gammaModel = glm!(double)(
        new Block1DParallel(), 
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new Adam!(double)(1E-6, 0.9, 0.999, 1E-6, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30), 
        totalCPUs, true, false);
  writeln("Adam Gradient Descent With Parallel Block Data \n", gammaModel);
}

/* Demo all the data types for AdaMax */
void gdAdaMaxDataDemo()
{
  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);
  
  /* Number of parameters */
  auto p = gammaX.ncol;
  
  auto gammaModel = glm!(double)(
        new Block1DParallel(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new GESVSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Full GLM Solve\n", gammaModel);

  writeln("The outputs for all these models should be the same.");
  gammaModel = glm!(double)(
        new RegularData(),
        gammaX, gammaY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new AdaMax!(double)(2E-2, 0.4, 0.999, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("AdaMax Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(
        new Block1D(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new AdaMax!(double)(2E-2, 0.4, 0.999, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("AdaMax Gradient Descent With Block Data \n", gammaModel);
  
  /* Gradient Descent Adam Block Model */
  gammaModel = glm!(double)(
        new Block1DParallel(), 
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new AdaMax!(double)(2E-2, 0.4, 0.999, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30), 
        totalCPUs, true, false);
  writeln("AdaMax Gradient Descent With Parallel Block Data \n", gammaModel);
}

/* Demo all the data types for NAdam */
void gdNAdamDataDemo()
{
  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);
  
  /* Number of parameters */
  auto p = gammaX.ncol;
  
  auto gammaModel = glm!(double)(
        new Block1DParallel(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new GESVSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Full GLM Solve\n", gammaModel);

  writeln("The outputs for all these models should be the same.");
  gammaModel = glm!(double)(
        new RegularData(),
        gammaX, gammaY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new NAdam!(double)(1E-2, 0.8, 0.999, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("NAdam Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(
        new Block1D(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new NAdam!(double)(1E-2, 0.8, 0.999, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("NAdam Gradient Descent With Block Data \n", gammaModel);
  
  /* Gradient Descent Adam Block Model */
  gammaModel = glm!(double)(
        new Block1DParallel(), 
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new NAdam!(double)(1E-2, 0.8, 0.999, 1E-8, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30), 
        totalCPUs, true, false);
  writeln("NAdam Gradient Descent With Parallel Block Data \n", gammaModel);
}


/* Demo all the data types for AMSGrad */
void gdAMSGradDataDemo()
{
  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;
  auto gammaBlockX = matrixToBlock(gammaX, 10);
  auto gammaBlockY = matrixToBlock(gammaY, 10);
  
  /* Number of parameters */
  auto p = gammaX.ncol;
  
  auto gammaModel = glm!(double)(
        new Block1DParallel(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new GESVSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Full GLM Solve\n", gammaModel);
  
  writeln("The outputs for all these models should be the same.");
  gammaModel = glm!(double)(
        new RegularData(),
        gammaX, gammaY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new AMSGrad!(double)(5E-3, 0.8, 0.999, 1E-6, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("AMSGrad Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(
        new Block1D(),
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new AMSGrad!(double)(5E-3, 0.8, 0.999, 1E-6, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30),
        true, false);
  writeln("AMSGrad Gradient Descent With Block Data \n", gammaModel);
  
  /* Gradient Descent Adam Block Model */
  gammaModel = glm!(double)(
        new Block1DParallel(), 
        gammaBlockX, gammaBlockY, 
        new GammaDistribution!(double)(), 
        new LogLink!(double)(),
        new AMSGrad!(double)(5E-3, 0.8, 0.999, 1E-6, p),
        new GETRIInverse!(double)(),
        new Control!(double)(30), 
        totalCPUs, true, false);
  writeln("AMSGrad Gradient Descent With Parallel Block Data \n", gammaModel);
}

/* Function contains many GLM examples */
void glm_demo()
{
  /* GLM Demo */

  /* Simulate the data */
  auto seed = 4;
  AbstractDistribution!(double) distrib = new GammaDistribution!(double)();
  AbstractLink!(double) link = new LogLink!(double)();
  auto gammaData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto gammaX = gammaData.X;
  auto gammaY = gammaData.y;

  distrib = new BinomialDistribution!(double)();
  link = new LogitLink!(double)();
  auto binomialData = simulateData!(double)(distrib, link, 30, 10_000, seed);
  auto binomialX = binomialData.X;
  auto binomialY = binomialData.y;

  distrib = new PoissonDistribution!(double)();
  link = new LogLink!(double)();
  auto poissonData = simulateData!(double)(distrib, link, 5, 100, seed);
  auto poissonX = poissonData.X;
  auto poissonY = poissonData.y;
  
  if(true)
  {
    /* Gamma Distribution With Log Link */
    auto gamma_distrib_log_link = glm(new RegularData(), gammaX, gammaY, 
        new GammaDistribution!double(), new LogLink!double());
    writeln(gamma_distrib_log_link);
    
    /* Gamma Distribution With Inverse Link */
    auto gamma_distrib_inv_link = glm(new RegularData(), gammaX, gammaY,
        new GammaDistribution!double(), new InverseLink!double());
    writeln(gamma_distrib_inv_link);
    
    /* Gamma Distribution With Identity Link */
    auto gamma_distrib_identity_link = glm(new RegularData(), gammaX, gammaY,
        new GammaDistribution!double(), new IdentityLink!double());
    writeln(gamma_distrib_identity_link);
    
    /* Gamma Distribution With Power Link */
    auto gamma_distrib_power_link = glm(new RegularData(), gammaX, gammaY,
        new GammaDistribution!double(), new PowerLink!double(0));
    writeln(gamma_distrib_power_link);
    auto gamma_distrib_power_link_2 = glm(new RegularData(), gammaX, gammaY,
        new GammaDistribution!double(), new PowerLink!double(1/3));
    writeln(gamma_distrib_power_link_2);
    
    /* Gamma Distribution With Negative Binomial Link */
    auto gamma_distrib_negative_binomial_link_1 = glm(new RegularData(), gammaX, gammaY,
        new GammaDistribution!double(), new NegativeBinomialLink!double(1.0));
    writeln(gamma_distrib_negative_binomial_link_1);
    auto gamma_distrib_negative_binomial_link_2 = glm(new RegularData(), gammaX, gammaY,
        new GammaDistribution!double(), new NegativeBinomialLink!double(2.0));
    writeln(gamma_distrib_negative_binomial_link_2);
    /* Binomial Distribution With Logit Link Function */
    auto binomial_logit_link = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new LogitLink!double());
    writeln(binomial_logit_link);
    openblas_set_num_threads(1); /* Set the number of BLAS threads */
    /* Binomial Distribution With Probit Link Function */
    auto binomial_probit_link = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new ProbitLink!double());
    writeln(binomial_probit_link);
    /* Binomial Distribution With CauchitLink Function */
    auto binomial_cauchit_link = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new CauchitLink!double());
    writeln(binomial_cauchit_link);
    /* Binomial Distribution With OddsPowerLink Function */
    auto binomial_distrib_odds_power_link_1 = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new OddsPowerLink!double(0.0));
    writeln(binomial_distrib_odds_power_link_1);
    auto binomial_distrib_odds_power_link_2 = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new OddsPowerLink!double(2.0));
    writeln(binomial_distrib_odds_power_link_2);
    
    //auto bernoulli_logcomplementary = glm(new RegularData(), binomialX, binomialY, 
    //    new BinomialDistribution!double(), new LogComplementLink!double());
    //writeln(bernoulli_logcomplementary);
    auto bernoulli_loglog = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new LogLogLink!double());
    writeln(bernoulli_loglog);
    auto bernoulli_complementaryloglog = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new ComplementaryLogLogLink!double());
    writeln(bernoulli_complementaryloglog);
    
    /* LogLink With Gaussian Distribution */
    auto log_link_gaussian_distrib = glm(new RegularData(), gammaX, gammaY, 
        new GaussianDistribution!double(), new LogLink!double());
    writeln(log_link_gaussian_distrib);
    
    auto log_link_gamma_distribution = glm(new RegularData(), gammaX, gammaY, 
        new GammaDistribution!double(), new LogLink!double());
    writeln(log_link_gamma_distribution);
    auto log_link_inversegaussian_distribution = glm(new RegularData(), gammaX, gammaY, 
        new InverseGaussianDistribution!double(), new LogLink!double());
    writeln(log_link_inversegaussian_distribution);
    auto logit_link_bernoulli_distrib = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new LogitLink!double());
    writeln(logit_link_bernoulli_distrib);
    auto log_link_negative_bernoulli_distrib = glm(new RegularData(), gammaX, gammaY, 
        new NegativeBinomialDistribution!double(0.5), new LogLink!double());
    writeln(log_link_negative_bernoulli_distrib);

    /* Commented out because it exceeds step control */
    //auto log_link_power_distrib = glm(new RegularData(), gammaX, gammaY, 
    //    new PowerDistribution!double(0.5), new PowerLink!double(0.5));
    //writeln(log_link_power_distrib);
    
    auto cauchit_link_binomial_distribution = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new CauchitLink!double());
    writeln(cauchit_link_binomial_distribution);
    
    /* Commented out because it exceeds step control */
    /* Binomial Distribution With LogComplementLink Function */
    //auto binomial_logcomplement_link = glm(new RegularData(), binomialX, binomialY, 
    //    new BinomialDistribution!double(), new LogComplementLink!double());
    //writeln(binomial_logcomplement_link);
    
    /* Binomial Distribution With LogLogLink Function */
    auto binomial_loglog_link = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new LogLogLink!double());
    writeln(binomial_loglog_link);
    /* Binomial Distribution With ComplementaryLogLogLink Function */
    auto binomial_complementaryloglog_link = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new ComplementaryLogLogLink!double());
    writeln(binomial_complementaryloglog_link);
  }
  if(false)
  {
    /* Now Test Different Distributions With Specific Link Functions */
    /* LogLink With Gaussian Distribution */
    auto log_link_gaussian_distrib = glm(new RegularData(), gammaX, gammaY, 
        new GaussianDistribution!double(), new LogLink!double());
    writeln(log_link_gaussian_distrib);
    /* LogLink With Gamma Distribution */
    auto log_link_gamma_distrib = glm(new RegularData(), gammaX, gammaY, 
        new GammaDistribution!double(), new LogLink!double());
    writeln(log_link_gamma_distrib);
    /* LogLink With InverseGaussian Distribution */
    auto log_link_inversegaussian_distrib = glm(new RegularData(), gammaX, gammaY, 
        new InverseGaussianDistribution!double(), new LogLink!double());
    writeln(log_link_inversegaussian_distrib);
    /* LogLink With Poisson Distribution */
    auto log_link_poisson_distrib = glm(new RegularData(), poissonX, poissonY, 
        new PoissonDistribution!double(), new LogLink!double());
    writeln(log_link_poisson_distrib);
    
    /* LogitLink With Binomial Distribution */
    auto logit_link_binomial_distrib = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new LogitLink!double());
    writeln(logit_link_binomial_distrib);
    /* LogLink With Negative Binomial Distribution */
    auto log_link_negative_binomial_distrib = glm(new RegularData(), gammaX, gammaY, 
        new NegativeBinomialDistribution!double(0.5), new LogLink!double());
    writeln(log_link_negative_binomial_distrib);
    /* LogLink With Power Distribution */
    auto log_link_power_distrib = glm(new RegularData(), gammaX, gammaY, 
        new PowerDistribution!double(0.5), new PowerLink!double(0.5));
    writeln(log_link_power_distrib);
    /* Logit Link With Binomial Distribution - Works fine */
    auto logit_link_binomial_distrib_two_col = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new LogitLink!double());
    writeln(logit_link_binomial_distrib_two_col);
    /* Cauchit Link With Binomial Distribution */
    auto cauchit_link_binomial_distrib_two_col = glm(new RegularData(), binomialX, binomialY, 
        new BinomialDistribution!double(), new CauchitLink!double());
    writeln(cauchit_link_binomial_distrib_two_col);
  }
}

