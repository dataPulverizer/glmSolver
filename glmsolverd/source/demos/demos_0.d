module demos.demo0;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;
import glmsolverd.link;
import glmsolverd.distributions;
import glmsolverd.tools;
import glmsolverd.linearalgebra;
import glmsolverd.io;
import glmsolverd.fit;

import std.conv: to;
import std.stdio : writeln;
import std.file: remove;
import std.parallelism;

/* ldc2 demos.d arrays.d arraycommon.d apply.d link.d distributions.d tools.d linearalgebra.d io.d fit.d -O2 -L-lopenblas -L-lpthread -L-llapacke -L-llapack -L-lm && ./demos */

void old_demo() /* Maybe remove this function in time */
{
  double[] dat1 = [0.5258874319129798,    0.1748310792322596, 0.32741218855864074, 
                   0.27457761265628555,   0.5884570435236942, 0.24725859282363394, 
                   0.0026890474662464303, 0.9497754886400656, 0.02207037565505421, 
                   0.6907347285327676,    0.9592865249385867, 0.0037546990281474013, 
                   0.5889903715624345,    0.9394951355167158, 0.4691847847916524, 
                   0.6715916314231278,    0.7554381258134812, 0.9471071671056135, 
                   0.5866722794791475,    0.8811154762774951];
  auto m1 = new Matrix!double(dat1, 5, 4);
  m1.writeln;
  int[] dat2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
  auto m2 = new Matrix!int(dat2, 5, 4);
  m2.writeln;

  double[] dat3 = [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                   11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
  auto m3 = new Matrix!double(dat3, 5, 4);
  m3.writeln;
  m3[2, 2] = 3.142;
  writeln("Change matrix index at m[2, 2]:\n", m3);
  //writeln("data:\n", m3.data);

  double[] dat4 = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9];
  auto v1 = new ColumnVector!double(dat4);
  v1.writeln;
  v1[3] = 3.142;
  writeln("Change index in the matrix:\n", v1);

  double[] dat5 = [0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9];
  auto v2 = new RowVector!double(dat5);
  v2.writeln;
  writeln("v2[3]: ", v2[3]);
  v2[3] = 3.142;
  writeln("Change index in the matrix:\n", v2);

  auto m6 = new Matrix!double([1.0, 2.0, 3.0, 4.0, 5.0, 
                     6.0, 7.0, 8.0, 9.0, 10.0,
                     11.0, 12.0, 13.0, 14.0, 15.0], 5, 3);
  auto v4 = new ColumnVector!double([1.0, 2.0, 3.0, 4.0, 5.0]);
  writeln("Sweep multiplication of \n", m6, "and\n", v4);
  writeln("Sweep multiplication of matrix and array \n", sweep!((double x1, double x2) => x1*x2, Column)(m6, v4.getData));

  writeln("Outcome of Column-wise matrix-vector multiplication sweep function\n", 
    sweep!((double x1, double x2) => x1*x2, Column)(m6, v4));
  auto v5 = new RowVector!double([1.0, 2.0, 3.0]);
  writeln("Outcome of Row-wise matrix-vector multiplcation sweep function\n", 
    sweep!((double x1, double x2) => x1*x2, Row)(m6, v5));
  
  auto m7 = new Matrix!double([16.0, 17.0, 18.0, 19.0, 20.0,
                               21.0, 22.0, 23.0, 24.0, 25.0,
                               26.0, 27.0, 28.0, 29.0, 30.0], 5, 3);
  writeln("Sweep function for two matrices\n", sweep!((double x1, double x2) => x1 * x2)(m6, m7));
  // double[5] arr1 = 1.0; //initialization example
  auto m8 = new Matrix!double([16.0, 17.0, 18.0, 19.0, 20.0,
                               21.0, 22.0, 23.0, 24.0, 25.0,
                               26.0, 27.0], 4, 3);
  
  /* This results in an error because the matrices have different dimensions */
  //writeln("This should be an error: ", sweep!((double x1, double x2) => x1 + x2)(m7, m8));

  /* Create a matrix using the array to mass on the matrix type */
  double[] arr = [1.0, 2.0, 3.0, 4.0];
  /* Testing the type inference for matrix constructor */
  auto m9 = matrix(arr, 2, 2);
  writeln("Type inferred constructed matrix: \n", m9);

  auto m10 = matrix(m9);

  writeln("Matrix multiplication: \n", mult_(m9, m10));
  writeln("Matrix multiplication: \n", mult_!(double, CblasColMajor, CblasNoTrans, CblasTrans)(m9, m10));

  auto m11 = matrix(m7);

  writeln("Original Matrix: ", m7);
  writeln("Transpose: ", m7.t());

  auto m13 = createRandomMatrix(5);
  writeln("Original Matrix: \n", m13);
  writeln("Transpose of square matrix: ", m13.t());

  auto v6 = columnVector([1.0, 2.0, 3.0]);

  auto v7 = mult_(m7, v6);
  writeln("Output of Matrix-Vector multiplication:\n", v7);

  auto v8 = columnVector([6.0, 7.0, 8.0, 9.0, 10.0]);
  writeln("Map function for column vector: \n", map!((double x1, double x2) => x1*x2)(v4, v8));

  auto v9 = rowVector([1.0, 2.0, 3.0, 4.0, 5.0]);
  auto v10 = rowVector([6.0, 7.0, 8.0, 9.0, 10.0]);
  writeln("Map function for row vector:\n", map!((double x1, double x2) => x1*x2)(v9, v10));

  writeln("Map function for column vector:\n", map!((double v) => v^^2)(v8));
  writeln("Map function for row vector:\n", map!((double v) => v^^2)(v9));

  auto m12 = createRandomMatrix(5);
  writeln("Create random square matrix:\n", m12);
  writeln("Inverse of a square matrix:\n", inv(m12));
  writeln("Pseudo-inverse of a square matrix:\n", pinv(m12));
  writeln("Create random rectangular matrix:\n", createRandomMatrix([cast(ulong)(7), cast(ulong)(3)]));

  writeln("Create random column vector:\n", createRandomColumnVector(5));
  writeln("Create random row vector:\n", createRandomRowVector(5));

  //auto sm1 = createSymmetricMatrix!double(9);
  double[] arr2 = [30, 1998, 1594, 1691, 1939, 2243, 1288, 1998, 138208, 108798, 
                   115325, 131824, 150101,  86673, 1594, 108798,  89036,  91903, 
                   104669, 119695,  69689, 1691, 115325,  91903,  99311, 111561, 
                   126821,  74462,   1939, 131824, 104669, 111561, 128459, 146097,  
                   85029, 2243, 150101, 119695, 126821, 146097, 170541,  97136, 
                   1288, 86673, 69689, 74462, 85029, 97136, 58368];
  auto sm1 = matrix(arr2, [7, 7]);
  writeln("Create random symmetric matrix:\n", sm1);
  writeln("General inverse of symmetric matrix:\n", inv(sm1));
  writeln("Symmetric inverse of symmetric matrix:\n", inv!(CblasSymmetric)(sm1));

  double[] arr3 = [477410, 32325450, 25832480, 27452590, 31399180, 36024970, 20980860];
  auto cv1 = columnVector(arr3);
  writeln("Matrix solve for general matrices: \n", solve(sm1, cv1));
  writeln("Matrix solve for symmetric matrices:\n", solve!(CblasSymmetric)(sm1, cv1));

  writeln("Epsilon: ", to!string(eps!(double)), ", Compliment Epsilon: ", to!string(ceps!(double)), "\n");
  writeln(new Control!(double)());

  writeln("Norm [1:6]:\n", norm([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
  
  auto v11 = columnVector(createRandomArray(5));
  auto v12 = columnVector(createRandomArray(5));

  writeln("Array 1: ", v11);
  writeln("Array 2: ", v12);
  writeln("absoluteError(ColumnVector1, ColumnVector2): ", absoluteError(v11, v12));
  writeln("relativeError(ColumnVector1, ColumnVector2): ", relativeError(v11, v12));

  writeln("Write this column vector to file:\n", v11);
  writeColumnVector("ColumnVector.bin", v11);
  auto v13 = readColumnVector!double("ColumnVector.bin");
  writeln("Read Column Vector from file:\n", v13);
  "ColumnVector.bin".remove();

  writeln("Write this row vector to file:\n", v9);
  writeRowVector("RowVector.bin", v9);
  auto v14 = readRowVector!double("RowVector.bin");
  writeln("Read Row Vector from file:\n", v14);
  "RowVector.bin".remove();

  auto m14 = createRandomMatrix([cast(ulong)(7), cast(ulong)(3)]);
  writeln("Matrix to be written to file:\n", m14);
  writeMatrix("Matrix.bin", m14);
  string xFile = "Matrix.bin";
  auto m15 = readMatrix!double(xFile);
  writeln("Matrix read from file:\n", m15);
  xFile.remove();

  return;
}

/* Will no longer be needed going forward - delete */
void qr_demo()
{
  auto X = matrix!double([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 
        1, 1, 1, 1, 1, 1, 1, 1], [20, 2]);
  auto y = columnVector!double([4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 
    5.17, 4.53, 5.33, 5.14, 4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 
    4.89, 4.32, 4.69]);
  auto qrOutput = qrls(X, y);
  writeln("QR decomposition Coefficient: ", qrOutput.coef);
  writeln("QR decomposition R: ", qrOutput.R);
}

/* Small demo may not be needed later */
void cars_glm_demo()
{
  /* Cars Data */
  string path = "/home/chib/code/GLMPrototype/";
  auto carsX = readMatrix!double(path ~ "data/carsX.bin");
  auto carsY = readMatrix!double(path ~ "data/carsY.bin");
  auto gamma_distrib_log_link_2 = glm(new RegularData(), carsX, carsY, 
    new GammaDistribution!double(), new LogLink!double(),
    new GESVSolver!(double)(), new GETRIInverse!(double)());
  writeln("Second Model:\n", gamma_distrib_log_link_2);
  writeln("Second Model Covariance Matrix:\n", gamma_distrib_log_link_2.cov);
}

/* Timed demo for basic observational benchmarking */
void timed_demo()
{
  /* GLM Demo */

  /* Data Load */
  string path = "/home/chib/code/GLMPrototype/";
  auto energyX = readMatrix!double(path ~ "data/energyX.bin");
  auto energyY = readMatrix!double(path ~ "data/energyY.bin");

  /* Gamma Distribution With Log Link */
  import std.datetime.stopwatch : AutoStart, StopWatch;
  openblas_set_num_threads(1); /* Set the number of cores used to 1 */
  auto sw = StopWatch(AutoStart.no);
  sw.start();
  auto gamma_distrib_log_link = glm(new RegularData(), energyX, energyY, 
      new GammaDistribution!(double)(), new LogLink!(double)(),
      new GESVSolver!(double)(), new GETRIInverse!(double)());
  sw.stop();
  writeln(gamma_distrib_log_link);
  writeln("Time taken: ", sw.peek.total!"msecs", " msec");

  return;
}

/* Testing single column matrix to column vector casts */
void testMatrixVectorConversions()
{
  auto mat1 = createRandomMatrix([cast(ulong)(10), cast(ulong)(1)]); // Column Matrix
  writeln("Column matrix: \n", mat1);

  auto vec = cast(ColumnVector!double)mat1;
  writeln("Converted to column vector: \n", vec);

  vec[0] = 99;
  writeln("Changed first item in the vector to 99, original: \n", mat1);

  /* Now cast to column vector */
  auto mat2 = cast(Matrix!(double))vec;
  writeln("Cast back to matrix from vector: ", mat2);

  /* Convert matrix to row vector */
  auto vec2 = cast(RowVector!(double))mat2;
  writeln("Cast matrix to row vector: ", vec2);
}

/* Read Block Demo */
import std.file : rmdirRecurse;
//import core.thread;
void blockMatrixDemo()
{
    string path = "/home/chib/code/glmSolver/data/testData/";
    auto blockMatrix = readBlockMatrix!(double)(path);
    writeln("Block Matrix: ", blockMatrix);
    string writePath = "/home/chib/code/glmSolver/data/writeBlockTest/";
    writeBlockMatrix!(double)(blockMatrix, writePath);
    //Thread.sleep(seconds(5));
    rmdirRecurse(writePath);
}

/* Testing link functions with block vectors */
void linkBlockDemo()
{
  auto link = new LogLink!(double)();
  auto mu = createRandomBlockColumnVector(20, 5);
  auto eta = createRandomBlockColumnVector(20, 5);
  writeln("Original mu:\n", mu);
  writeln("Original eta:\n", eta);
  writeln("Link function test on block vectors:\n", link.linkfun(mu));
  writeln("Inverse link function test on block vectors:\n", link.linkinv(eta));
  writeln("deta_dmu function test on block vectors:\n", link.deta_dmu(mu, eta));
}

void blockGLMDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyX = readBlockMatrix!(double)(path ~ "energyBlockX/");
  auto energyY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  auto blockGLMModel = glm!(double)(new Block1D(), energyX, 
        energyY, new GaussianDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Block Model\n", blockGLMModel);
  
  auto energyX2 = readMatrix!(double)(path ~ "energyX.bin");
  auto energyY2 = readMatrix!(double)(path ~ "energyY.bin");
  auto glmModel = glm!(double)(new RegularData(), energyX2, 
        energyY2, new GaussianDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Regular Model\n", glmModel);
  
  auto educationBlockX = readBlockMatrix!(double)(path ~ "educationBlockX/");
  auto educationBlockY = readBlockMatrix!(double)(path ~ "educationBlockY/");
  auto eduBlockModel = glm!(double)(new Block1D(), educationBlockX, 
        educationBlockY, new BinomialDistribution!(double)(), 
        new LogitLink!(double)(), new VanillaSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Block Model\n", eduBlockModel);

  auto educationX = readMatrix!(double)(path ~ "educationX.bin");
  auto educationY = readMatrix!(double)(path ~ "educationY.bin");
  auto eduModel = glm!(double)(new RegularData(), educationX, 
        educationY, new BinomialDistribution!(double)(), 
        new LogitLink!(double)(), new VanillaSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Regular Model\n", eduModel);

  auto gammaBlockModel = glm!(double)(new Block1D(), energyX, 
        energyY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Block Model\n", gammaBlockModel);
  
  auto gammaModel = glm!(double)(new RegularData(), energyX2, 
        energyY2, new GammaDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Regular Model\n", gammaModel);
}

/* Compares block and parallel block algorithm output */
void testParallel()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");

  /* For credit data - Binomial data */
  auto creditBlockX = readBlockMatrix!(double)(path ~ "creditBlockX/");
  auto creditBlockY = readBlockMatrix!(double)(path ~ "creditBlockY/");

  if(false)
  {
    auto blockGLMModel = glm!(double)(new Block1D(), energyBlockX, 
          energyBlockY, new GaussianDistribution!(double)(), new LogLink!(double)(),
          new SYSVSolver!(double)(), new GETRIInverse!(double)());
    writeln("Block Model\n", blockGLMModel);
    
    auto blockParallelGLMModel = glm!(double)(new Block1DParallel(),
          energyBlockX, energyBlockY, new GaussianDistribution!(double)(),
          new LogLink!(double)(), new SYSVSolver!(double)(),
          new GETRIInverse!(double)());
    writeln("Parallel Block Model\n", blockParallelGLMModel);
  }
  
  auto blockGLMModel = glm!(double)(new Block1D(), creditBlockX, 
        creditBlockY, new BinomialDistribution!(double)(), new LogitLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Block Model\n", blockGLMModel);

  auto blockParallelGLMModel = glm!(double)(new Block1DParallel(), creditBlockX, 
        creditBlockY, new BinomialDistribution!(double)(), new LogitLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Parallel Block Model\n", blockParallelGLMModel);
}


void parallelBlockGLMDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  auto creditBlockX = readBlockMatrix!(double)(path ~ "creditBlockX/");
  auto creditBlockY = readBlockMatrix!(double)(path ~ "creditBlockY/");
  auto educationBlockX = readBlockMatrix!(double)(path ~ "educationBlockX/");
  auto educationBlockY = readBlockMatrix!(double)(path ~ "educationBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");
  auto creditX = readMatrix!(double)(path ~ "creditX.bin");
  auto creditY = readMatrix!(double)(path ~ "creditY.bin");
  auto educationX = readMatrix!(double)(path ~ "educationX.bin");
  auto educationY = readMatrix!(double)(path ~ "educationY.bin");

  auto blockGLMModel = glm!(double)(new Block1D(), energyBlockX, 
        energyBlockY, new GaussianDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Block Model\n", blockGLMModel);
  
  auto blockParallelGLMModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GaussianDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Parallel Block Model\n", blockParallelGLMModel);
  
  auto glmModel = glm!(double)(new RegularData(), energyX, 
        energyY, new GaussianDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Regular Model\n", glmModel);

  /***************************************************************/
  /* Identity Link for testing canonical */
  blockGLMModel = glm!(double)(new Block1D(), energyBlockX, 
        energyBlockY, new GaussianDistribution!(double)(), new IdentityLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Block Model\n", blockGLMModel);
  
  blockParallelGLMModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GaussianDistribution!(double)(), new IdentityLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Parallel Block Model\n", blockParallelGLMModel);

  glmModel = glm!(double)(new RegularData(), energyX, 
        energyY, new GaussianDistribution!(double)(), new IdentityLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Regular Model\n", glmModel);
  /***************************************************************/
  /* Credit Dataset */
  blockGLMModel = glm!(double)(new Block1D(), creditBlockX, 
        creditBlockY, new BinomialDistribution!(double)(), new LogitLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Block Model\n", blockGLMModel);
  
  blockParallelGLMModel = glm!(double)(new Block1DParallel(), creditBlockX, 
        creditBlockY, new BinomialDistribution!(double)(), new LogitLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Parallel Block Model\n", blockParallelGLMModel);

  glmModel = glm!(double)(new RegularData(), creditX, 
        creditY, new BinomialDistribution!(double)(), new LogitLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Regular Model\n", glmModel);
  /***************************************************************/
  auto eduBlockModel = glm!(double)(new Block1D(), educationBlockX, 
        educationBlockY, new BinomialDistribution!(double)(), 
        new LogitLink!(double)(), new VanillaSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Block Model\n", eduBlockModel);

  auto eduParallelBlockModel = glm!(double)(new Block1DParallel(), educationBlockX, 
        educationBlockY, new BinomialDistribution!(double)(), 
        new LogitLink!(double)(), new VanillaSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Parallel Block Model\n", eduParallelBlockModel);

  auto eduModel = glm!(double)(new RegularData(), educationX, 
        educationY, new BinomialDistribution!(double)(), 
        new LogitLink!(double)(), new VanillaSolver!(double)(), 
        new GETRIInverse!(double)());
  writeln("Regular Model\n", eduModel);

  auto gammaBlockModel = glm!(double)(new Block1D(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Block Model\n", gammaBlockModel);

  auto gammaParallelBlockModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Parallel Block Model\n", gammaParallelBlockModel);
  
  auto gammaModel = glm!(double)(new RegularData(), energyX, 
        energyY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Regular Model\n", gammaModel);
  /***************************************************************/
}


void gdMomentumDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");

  /* Number of parameters */
  auto p = energyBlockX[0].ncol;
  
  auto gammaModel = glm!(double)(new RegularData(), energyX, 
        energyY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Regular Model\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GradientDescent!(double)(1E-5), new GETRIInverse!(double)(),
        new Control!(double)(10));
  writeln("Gradient Descent solver with parallel data \n", gammaModel);

  /* Gradient Descent Momentum Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Momentum!(double)(1E-5, 0.80, p),
        new GETRIInverse!(double)(), new Control!(double)(10));
  writeln("Gradient Descent solver with parallel data for Momentum Solver \n", gammaModel);
}

/* Test all the data types for momentum */
void gdMomentumDataDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");

  /* Number of parameters */
  auto p = energyBlockX[0].ncol;

  writeln("The outputs for all these models should be the same.");
  auto gammaModel = glm!(double)(new RegularData(), energyX, 
        energyY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Momentum!(double)(1E-5, 0.9, p), new GETRIInverse!(double)(),
        new Control!(double)(10));
  writeln("Momentum Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Momentum!(double)(1E-5, 0.9, p), new GETRIInverse!(double)(),
        new Control!(double)(10));
  writeln("Momentum Gradient Descent With Parallel Data \n", gammaModel);

  /* Gradient Descent Momentum Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Momentum!(double)(1E-5, 0.9, p),
        new GETRIInverse!(double)(), new Control!(double)(10));
  writeln("Momentum Gradient Descent With Parallel Data \n", gammaModel);
}

/* Test all the data types for nesterov */
void gdNesterovDataDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyScaledBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyScaledY.bin");

  /* Number of parameters */
  auto p = energyBlockX[0].ncol;

  auto gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Full GLM Solve\n", gammaModel);

  writeln("The outputs for all these models should be the same.");
  gammaModel = glm!(double)(new RegularData(), energyX, 
        energyY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(3E-7, 0.7, p), new GETRIInverse!(double)(),
        new Control!(double)(10), true, false);
  writeln("Nesterov Gradient Descent With Regular Data\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(new Block1D(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(3E-7, 0.7, p), new GETRIInverse!(double)(),
        new Control!(double)(10), true, false);
  writeln("Nesterov Gradient Descent With Block Data \n", gammaModel);
  
  /* Gradient Descent Nesterov Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(3E-7, 0.7, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true, false);
  writeln("Nesterov Gradient Descent With Parallel Block Data \n", gammaModel);
}

void gdNesterovVsMomentumDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");

  /* Number of parameters */
  auto p = energyBlockX[0].ncol;

  /***************************************************************/
  /* Gradient Descent Block Model */
  auto gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Momentum!(double)(1E-6, 0.6, p), new GETRIInverse!(double)(),
        new Control!(double)(10), true);
  writeln("Momentum Gradient Descent With Parallel Block Data \n", gammaModel);

  /* Gradient Descent Nesterov Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.6, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("Nesterov Gradient Descent With Parallel Block Data \n", gammaModel);
}

void gdNesterovVsAdagradDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");

  /* Number of parameters */
  auto p = energyBlockX[0].ncol;

  /***************************************************************/
  /* Gradient Descent Nesterov Block Model */
  auto gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.9, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("Nesterov Gradient Descent With Parallel Block Data \n", gammaModel);

  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Adagrad!(double)(1E-6, 1E-6, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("Adagrad Gradient Descent With Parallel Block Data \n", gammaModel);
}


void gdNesterovVsAdadeltaDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");

  /* Number of parameters */
  auto p = energyBlockX[0].ncol;

  /***************************************************************/
  /* Gradient Descent Nesterov Block Model */
  auto gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.9, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("Nesterov Gradient Descent With Parallel Block Data \n", gammaModel);

  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Adadelta!(double)(0.9, 1E-6, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("Adadelta Gradient Descent With Parallel Block Data \n", gammaModel);
}


void gdNesterovVsRMSpropDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");

  /* Number of parameters */
  auto p = energyBlockX[0].ncol;

  /***************************************************************/
  /* Gradient Descent Nesterov Block Model */
  auto gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.9, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("Nesterov Gradient Descent With Parallel Block Data \n", gammaModel);

  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new RMSprop!(double)(0.9, 1E-6, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("RMSProp Gradient Descent With Parallel Block Data \n", gammaModel);
}

void gdNesterovVsAdamDemo()
{
  string path = "/home/chib/code/glmSolver/data/";
  
  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");
  
  /* Number of parameters */
  auto p = energyBlockX[0].ncol;
  
  /***************************************************************/
  /* Gradient Descent Nesterov Block Model */
  auto gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.9, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("Nesterov Gradient Descent With Parallel Block Data \n", gammaModel);
  
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Adam!(double)(1E-6, 0.9, 0.999, 1E-6, p),
        new GETRIInverse!(double)(), new Control!(double)(100), 
        totalCPUs, true);
  writeln("Adam Gradient Descent With Parallel Block Data \n", gammaModel);
}

void gdNesterovVsAdaMaxDemo()
{
  string path = "/home/chib/code/glmSolver/data/";
  
  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");
  
  /* Number of parameters */
  auto p = energyBlockX[0].ncol;
  
  /***************************************************************/
  /* Gradient Descent Nesterov Block Model */
  auto gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.9, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("Nesterov Gradient Descent With Parallel Block Data \n", gammaModel);
  
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new AdaMax!(double)(1E-6, 0.9, 0.999, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, false);
  writeln("Adam Gradient Descent With Parallel Block Data \n", gammaModel);
}

void gdNesterovVsAdamDemo()
{
  string path = "/home/chib/code/glmSolver/data/";
  
  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");
  
  /* Number of parameters */
  auto p = energyBlockX[0].ncol;
  
  /***************************************************************/
  /* Gradient Descent Nesterov Block Model */
  auto gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.9, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("Nesterov Gradient Descent With Parallel Block Data \n", gammaModel);
  
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Adam!(double)(1E-6, 0.9, 0.999, 1E-6, p),
        new GETRIInverse!(double)(), new Control!(double)(100), 
        totalCPUs, true);
  writeln("Adam Gradient Descent With Parallel Block Data \n", gammaModel);
}

void gdNesterovVsNAdamDemo()
{
  string path = "/home/chib/code/glmSolver/data/";
  
  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");
  
  /* Number of parameters */
  auto p = energyBlockX[0].ncol;
  
  /***************************************************************/
  /* Gradient Descent Nesterov Block Model */
  auto gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(1E-6, 0.9, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("Nesterov Gradient Descent With Parallel Block Data \n", gammaModel);
  
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new NAdam!(double)(1E-2, 0.9, 0.999, 1E-6, p),
        new GETRIInverse!(double)(), new Control!(double)(10), 
        totalCPUs, true);
  writeln("NAdam Gradient Descent With Parallel Block Data \n", gammaModel);
}

void gdActualVsAMSGradDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyScaledBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyScaledY.bin");
  
  /* Number of parameters */
  auto p = energyBlockX[0].ncol;
  
  /***************************************************************/
  /* Gradient Descent Nesterov Block Model */
  auto gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Full GLM Solve\n", gammaModel);
  
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new AMSGrad!(double)(5E-3, 0.8, 0.999, 1E-6, p),
        new GETRIInverse!(double)(), new Control!(double)(50), 
        totalCPUs, /* calculate covariance */ true,
        /* step control */ true);
  writeln("AMSGrad Gradient Descent With Parallel Block Data \n", gammaModel);
}



void normDemo()
{
  /* Quick test to see if norm is functioning correctly */
  //auto x = new ColumnVector!float([1.0f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f]);
  auto x = new ColumnVector!double([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  writeln("Norm demo: ", norm(x), "\n");
}

void gradientDescentGLMDemo()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto energyBlockX = readBlockMatrix!(double)(path ~ "energyScaledBlockX/");
  auto energyBlockY = readBlockMatrix!(double)(path ~ "energyBlockY/");
  
  auto energyX = readMatrix!(double)(path ~ "energyScaledX.bin");
  auto energyY = readMatrix!(double)(path ~ "energyY.bin");

  /* Number of parameters */
  auto p = energyBlockX[0].ncol;
  
  auto gammaModel = glm!(double)(new RegularData(), energyX, 
        energyY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new VanillaSolver!(double)(), new GETRIInverse!(double)());
  writeln("Regular Model\n", gammaModel);
  /***************************************************************/
  /* Gradient Descent */
  gammaModel = glm!(double)(new RegularData(), energyX, 
        energyY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GradientDescent!(double)(1E-4), new GETRIInverse!(double)(),
        new Control!(double)(10));
  writeln("Gradient Descent solver with regular data \n", gammaModel);
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(new Block1D(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GradientDescent!(double)(1E-4), new GETRIInverse!(double)(),
        new Control!(double)(10));
  writeln("Gradient Descent solver with block data \n", gammaModel);
  /* Gradient Descent Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new GradientDescent!(double)(1E-4), new GETRIInverse!(double)(),
        new Control!(double)(10));
  writeln("Gradient Descent solver with parallel data \n", gammaModel);

  /* Gradient Descent Momentum Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Momentum!(double)(5E-6, 0.80, p),
        new GETRIInverse!(double)(), new Control!(double)(10));
  writeln("Gradient Descent solver with parallel data for Momentum Solver \n", gammaModel);

  /* Gradient Descent Nesterov Block Model */
  gammaModel = glm!(double)(new Block1DParallel(), energyBlockX, 
        energyBlockY, new GammaDistribution!(double)(), new LogLink!(double)(),
        new Nesterov!(double)(5E-6, 0.80, p),
        new GETRIInverse!(double)(), new Control!(double)(10));
  writeln("Gradient Descent solver with parallel data for Nesterov Solver \n", gammaModel);
}


void testBlockIO()
{
  string path = "/home/chib/code/glmSolver/data/";

  auto carsX = readMatrix!(double)(path ~ "carsScaledX.bin");
  auto carsBlockX = readBlockMatrix!(double)(path ~ "carsScaledBlockX/");

  writeln("Cars X: ", carsX);
  writeln("Cars X block: ", carsBlockX);
}


/* Test distribution function with block vectors and matrices */
void distribBlockDemo()
{
  auto distrib = new GaussianDistribution!(double)();
  auto yM = createRandomBlockMatrix!(double)(10, 1, 5);
  ColumnVector!(double)[] wts;
  auto tpl = distrib.init(yM, wts);
  auto y = tpl[0];
  auto mu = tpl[1];
  auto var = distrib.variance(mu);
  auto dev = distrib.devianceResiduals(mu, y);
  writeln("y:\n", y);
  writeln("mu:\n", mu);
  writeln("var:\n", var);
  writeln("dev:\n", dev);
}


/* Function contains many GLM examples */
void glm_demo()
{
  /* GLM Demo */

  /* Data Load */
  string path = "/home/chib/code/GLMPrototype/";
  auto energyX = readMatrix!double(path ~ "data/energyX.bin");
  auto energyY = readMatrix!double(path ~ "data/energyY.bin");

  /* Insurance data */
  auto insuranceX = readMatrix!double(path ~ "data/insuranceX.bin");
  auto insuranceY = readMatrix!double(path ~ "data/insuranceY.bin");
  
  /* Credit Card Fraud */
  auto creditX = readMatrix!double(path ~ "data/creditX.bin");
  auto creditY = readMatrix!double(path ~ "data/creditY.bin");
  
  /* GPA Data */
  auto gpaX = readMatrix!double(path ~ "data/gpaX.bin");
  auto gpaY = readMatrix!double(path ~ "data/gpaY.bin");
  
  /* Cars Data */
  auto carsX = readMatrix!double(path ~ "data/carsX.bin");
  auto carsY = readMatrix!double(path ~ "data/carsY.bin");
  
  /* Quine Data for negative Binomial Distribution */
  auto quineX = readMatrix!double(path ~ "data/quineX.bin");
  auto quineY = readMatrix!double(path ~ "data/quineY.bin");

  /* Education Data for negative Binomial Distribution */
  auto educationX = readMatrix!double(path ~ "data/educationX.bin");
  auto educationY = readMatrix!double(path ~ "data/educationY.bin");
  
  if(true)
  {
    /* Gamma Distribution With Log Link */
    auto gamma_distrib_log_link = glm(new RegularData(), energyX, energyY, 
        new GammaDistribution!double(), new LogLink!double());
    writeln(gamma_distrib_log_link);
    
    /* Gamma Distribution With Inverse Link */
    auto gamma_distrib_inv_link = glm(new RegularData(), energyX, energyY,
        new GammaDistribution!double(), new InverseLink!double());
    writeln(gamma_distrib_inv_link);
    
    /* Gamma Distribution With Identity Link */
    auto gamma_distrib_identity_link = glm(new RegularData(), energyX, energyY,
        new GammaDistribution!double(), new IdentityLink!double());
    writeln(gamma_distrib_identity_link);
    
    /* Gamma Distribution With Power Link */
    auto gamma_distrib_power_link = glm(new RegularData(), energyX, energyY,
        new GammaDistribution!double(), new PowerLink!double(0));
    writeln(gamma_distrib_power_link);
    auto gamma_distrib_power_link_2 = glm(new RegularData(), carsX, carsY,
        new GammaDistribution!double(), new PowerLink!double(1/3));
    writeln(gamma_distrib_power_link_2);
    
    /* Gamma Distribution With Negative Binomial Link */
    auto gamma_distrib_negative_binomial_link_1 = glm(new RegularData(), carsX, carsY,
        new GammaDistribution!double(), new NegativeBinomialLink!double(1.0));
    writeln(gamma_distrib_negative_binomial_link_1);
    auto gamma_distrib_negative_binomial_link_2 = glm(new RegularData(), energyX, energyY,
        new GammaDistribution!double(), new NegativeBinomialLink!double(2.0));
    writeln(gamma_distrib_negative_binomial_link_2);
    /* Binomial Distribution With Logit Link Function */
    auto binomial_logit_link = glm(new RegularData(), creditX, creditY, 
        new BinomialDistribution!double(), new LogitLink!double());
    writeln(binomial_logit_link);
    openblas_set_num_threads(1); /* Set the number of BLAS threads */
    /* Binomial Distribution With Probit Link Function */
    auto binomial_probit_link = glm(new RegularData(), gpaX, gpaY, 
        new BinomialDistribution!double(), new ProbitLink!double());
    writeln(binomial_probit_link);
    /* Binomial Distribution With CauchitLink Function */
    auto binomial_cauchit_link = glm(new RegularData(), gpaX, gpaY, 
        new BinomialDistribution!double(), new CauchitLink!double());
    writeln(binomial_cauchit_link);
    /* Binomial Distribution With OddsPowerLink Function */
    auto binomial_oddspower_link = glm(new RegularData(), educationX, educationY, 
        new BinomialDistribution!double(), new OddsPowerLink!double(2));
    writeln(binomial_oddspower_link);
  
    auto binomial_distrib_odds_power_link_1 = glm(new RegularData(), creditX, creditY, 
        new BinomialDistribution!double(), new OddsPowerLink!double(0.0));
    writeln(binomial_distrib_odds_power_link_1);
    auto binomial_distrib_odds_power_link_2 = glm(new RegularData(), educationX, educationY, 
        new BinomialDistribution!double(), new OddsPowerLink!double(2.0));
    writeln(binomial_distrib_odds_power_link_2);
    
    auto bernoulli_logcomplementary = glm(new RegularData(), gpaX, gpaY, 
        new BinomialDistribution!double(), new LogComplementLink!double());
    writeln(bernoulli_logcomplementary);
    auto bernoulli_loglog = glm(new RegularData(), gpaX, gpaY, 
        new BinomialDistribution!double(), new LogLogLink!double());
    writeln(bernoulli_loglog);
    auto bernoulli_complementaryloglog = glm(new RegularData(), gpaX, gpaY, 
        new BinomialDistribution!double(), new ComplementaryLogLogLink!double());
    writeln(bernoulli_complementaryloglog);
    
    /* LogLink With Gaussian Distribution */
    auto log_link_gaussian_distrib = glm(new RegularData(), energyX, energyY, 
        new GaussianDistribution!double(), new LogLink!double());
    writeln(log_link_gaussian_distrib);
    
    auto log_link_gamma_distribution = glm(new RegularData(), energyX, energyY, 
        new GammaDistribution!double(), new LogLink!double());
    writeln(log_link_gamma_distribution);
    auto log_link_inversegaussian_distribution = glm(new RegularData(), energyX, energyY, 
        new InverseGaussianDistribution!double(), new LogLink!double());
    writeln(log_link_inversegaussian_distribution);
    auto log_link_poisson_distribution = glm(new RegularData(), energyX, energyY, 
        new PoissonDistribution!double(), new LogLink!double());
    writeln(log_link_poisson_distribution);
    auto logit_link_bernoulli_distrib = glm(new RegularData(), creditX, creditY, 
        new BinomialDistribution!double(), new LogitLink!double());
    writeln(logit_link_bernoulli_distrib);
    auto log_link_negative_bernoulli_distrib = glm(new RegularData(), energyX, energyY, 
        new NegativeBinomialDistribution!double(0.5), new LogLink!double());
    writeln(log_link_negative_bernoulli_distrib);
    auto log_link_power_distrib = glm(new RegularData(), carsX, carsY, 
        new PowerDistribution!double(0.5), new PowerLink!double(0.5));
    writeln(log_link_power_distrib);
    auto logit_link_binomial_distribution = glm(new RegularData(), educationX, educationY, 
        new BinomialDistribution!double(), new LogLink!double());
    writeln(logit_link_binomial_distribution);
    auto cauchit_link_binomial_distribution = glm(new RegularData(), educationX, educationY, 
        new BinomialDistribution!double(), new CauchitLink!double());
    writeln(cauchit_link_binomial_distribution);
  }
  if(false)
  {
      /* Binomial Distribution With LogComplementLink Function */
    auto binomial_logcomplement_link = glm(new RegularData(), gpaX, gpaY, 
        new BinomialDistribution!double(), new LogComplementLink!double());
    writeln(binomial_logcomplement_link);
    /* Binomial Distribution With LogLogLink Function */
    auto binomial_loglog_link = glm(new RegularData(), gpaX, gpaY, 
        new BinomialDistribution!double(), new LogLogLink!double());
    writeln(binomial_loglog_link);
    /* Binomial Distribution With ComplementaryLogLogLink Function */
    auto binomial_complementaryloglog_link = glm(new RegularData(), gpaX, gpaY, 
        new BinomialDistribution!double(), new ComplementaryLogLogLink!double());
    writeln(binomial_complementaryloglog_link);
    
    /* Now Test Different Distributions With Specific Link Functions */
    /* LogLink With Gaussian Distribution */
    auto log_link_gaussian_distrib = glm(new RegularData(), energyX, energyY, 
        new GaussianDistribution!double(), new LogLink!double());
    writeln(log_link_gaussian_distrib);
    /* LogLink With Gamma Distribution */
    auto log_link_gamma_distrib = glm(new RegularData(), energyX, energyY, 
        new GammaDistribution!double(), new LogLink!double());
    writeln(log_link_gamma_distrib);
    /* LogLink With InverseGaussian Distribution */
    auto log_link_inversegaussian_distrib = glm(new RegularData(), energyX, energyY, 
        new InverseGaussianDistribution!double(), new LogLink!double());
    writeln(log_link_inversegaussian_distrib);
    /* LogLink With Poisson Distribution */
    auto log_link_poisson_distrib = glm(new RegularData(), energyX, energyY, 
        new PoissonDistribution!double(), new LogLink!double());
    writeln(log_link_poisson_distrib);
    
    /* LogitLink With Binomial Distribution */
    auto logit_link_binomial_distrib = glm(new RegularData(), creditX, creditY, 
        new BinomialDistribution!double(), new LogitLink!double());
    writeln(logit_link_binomial_distrib);
    /* LogitLink With Negative Binomial Distribution */
    auto logit_link_negative_binomial_distrib = glm(new RegularData(), energyX, energyY, 
        new NegativeBinomialDistribution!double(0.5), new LogLink!double());
    writeln(logit_link_negative_binomial_distrib);
    /* LogLink With Power Distribution */
    auto log_link_power_distrib = glm(new RegularData(), carsX, carsY, 
        new PowerDistribution!double(0.5), new PowerLink!double(0.5));
    writeln(log_link_power_distrib);
    /* Logit Link With Binomial Distribution - Works fine */
    auto logit_link_binomial_distrib_two_col = glm(new RegularData(), educationX, educationY, 
        new BinomialDistribution!double(), new LogitLink!double());
    writeln(logit_link_binomial_distrib_two_col);
    /* Cauchit Link With Binomial Distribution */
    auto cauchit_link_binomial_distrib_two_col = glm(new RegularData(), educationX, educationY, 
        new BinomialDistribution!double(), new CauchitLink!double());
    writeln(cauchit_link_binomial_distrib_two_col);
  }
}

