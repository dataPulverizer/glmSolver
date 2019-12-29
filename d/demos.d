import arrays;
import arraycommon;
import apply;
import link;
import distributions;
import tools;
import linearalgebra;
import io;
import fit;
import std.conv: to;
import std.stdio : writeln;
import std.file: remove;

/* ldc2 demos.d arrays.d arraycommon.d apply.d link.d distributions.d tools.d linearalgebra.d io.d fit.d -O2 -L-lopenblas -L-lpthread -L-llapacke -L-llapack -L-lm && ./demos */

void demo1()
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
  writeln("Create random rectangular matrix:\n", createRandomMatrix(7, 3));

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

  auto m14 = createRandomMatrix(7, 3);
  writeln("Matrix to be written to file:\n", m14);
  writeMatrix("Matrix.bin", m14);
  string xFile = "Matrix.bin";
  auto m15 = readMatrix!double(xFile);
  writeln("Matrix read from file:\n", m15);
  xFile.remove();

  return;
}

void qr_test()
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

void qr_vs_conventional()
{
  /* GLM Demo */

  /* Data Load */
  string path = "/home/chib/code/GLMPrototype/";
  auto energyX = readMatrix!double(path ~ "data/energyX.bin");
  auto energyY = readMatrix!double(path ~ "data/energyY.bin");

  /* Gamma Distribution With Log Link */
  import std.datetime.stopwatch : AutoStart, StopWatch;
  auto sw = StopWatch(AutoStart.no);
  sw.start();
  auto gamma_distrib_log_link = glm(energyX, energyY, 
      new GammaDistribution!double(), new LogLink!double());
  sw.stop();
  writeln(gamma_distrib_log_link);
  writeln("Time taken: ", sw.peek.total!"msecs");

  return;
}

void testMatrixVectorConversions()
{
  auto mat1 = createRandomMatrix(10, 1); // Column Matrix
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

void main()
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

  if(false)
  {
  /* Gamma Distribution With Log Link */
  auto gamma_distrib_log_link = glm(energyX, energyY, new GammaDistribution!double(), new LogLink!double());
  writeln(gamma_distrib_log_link);
  
  /* Gamma Distribution With Inverse Link */
  auto gamma_distrib_inv_link = glm(energyX, energyY, new GammaDistribution!double(), new InverseLink!double());
  writeln(gamma_distrib_inv_link);
  
  /* Gamma Distribution With Identity Link */
  auto gamma_distrib_identity_link = glm(energyX, energyY, new GammaDistribution!double(), new IdentityLink!double());
  writeln(gamma_distrib_identity_link);
  
  /* Gamma Distribution With Power Link */
  auto gamma_distrib_power_link = glm(energyX, energyY, new GammaDistribution!double(), new PowerLink!double(0));
  writeln(gamma_distrib_power_link);
  auto gamma_distrib_power_link_2 = glm(carsX, carsY, new GammaDistribution!double(), new PowerLink!double(1/3));
  writeln(gamma_distrib_power_link_2);

  /* Gamma Distribution With Negative Binomial Link */
  auto gamma_distrib_negative_binomial_link_1 = glm(carsX, carsY, new GammaDistribution!double(), new NegativeBinomialLink!double(1.0));
  writeln(gamma_distrib_negative_binomial_link_1);
  auto gamma_distrib_negative_binomial_link_2 = glm(energyX, energyY, new GammaDistribution!double(), new NegativeBinomialLink!double(2.0));
  writeln(gamma_distrib_negative_binomial_link_2);
  /* Binomial Distribution With Logit Link Function */
  auto binomial_logit_link = glm(creditX, creditY, 
      new BinomialDistribution!double(), new LogitLink!double());
  writeln(binomial_logit_link);
  openblas_set_num_threads(1); /* Set the number of BLAS threads */
  /* Binomial Distribution With Probit Link Function */
  auto binomial_probit_link = glm(gpaX, gpaY, 
      new BinomialDistribution!double(), new ProbitLink!double());
  writeln(binomial_probit_link);
  /* Binomial Distribution With CauchitLink Function */
  auto binomial_cauchit_link = glm(gpaX, gpaY, 
      new BinomialDistribution!double(), new CauchitLink!double());
  writeln(binomial_cauchit_link);
  /* Binomial Distribution With OddsPowerLink Function */
  auto binomial_oddspower_link = glm(creditX, creditY, 
      new BinomialDistribution!double(), new OddsPowerLink!double(1));
  writeln(binomial_oddspower_link);
  /* Binomial Distribution With LogComplementLink Function */
  auto binomial_logcomplement_link = glm(gpaX, gpaY, 
      new BinomialDistribution!double(), new LogComplementLink!double());
  writeln(binomial_logcomplement_link);
  /* Binomial Distribution With LogLogLink Function */
  auto binomial_loglog_link = glm(gpaX, gpaY, 
      new BinomialDistribution!double(), new LogLogLink!double());
  writeln(binomial_loglog_link);
  /* Binomial Distribution With ComplementaryLogLogLink Function */
  auto binomial_complementaryloglog_link = glm(gpaX, gpaY, 
      new BinomialDistribution!double(), new ComplementaryLogLogLink!double());
  writeln(binomial_complementaryloglog_link);

  /* Now Test Different Distributions With Specific Link Functions */
  /* LogLink With Gaussian Distribution */
  auto log_link_gaussian_distrib = glm(energyX, energyY, 
      new GaussianDistribution!double(), new LogLink!double());
  writeln(log_link_gaussian_distrib);
  /* LogLink With Gamma Distribution */
  auto log_link_gamma_distrib = glm(energyX, energyY, 
      new GammaDistribution!double(), new LogLink!double());
  writeln(log_link_gamma_distrib);
  /* LogLink With InverseGaussian Distribution */
  auto log_link_inversegaussian_distrib = glm(energyX, energyY, 
      new InverseGaussianDistribution!double(), new LogLink!double());
  writeln(log_link_inversegaussian_distrib);
  /* LogLink With Poisson Distribution */
  auto log_link_poisson_distrib = glm(energyX, energyY, 
      new PoissonDistribution!double(), new LogLink!double());
  writeln(log_link_poisson_distrib);

  /* LogitLink With Binomial Distribution */
  auto logit_link_binomial_distrib = glm(creditX, creditY, 
      new BinomialDistribution!double(), new LogitLink!double());
  writeln(logit_link_binomial_distrib);
  /* LogitLink With Negative Binomial Distribution */
  auto logit_link_negative_binomial_distrib = glm(energyX, energyY, 
      new NegativeBinomialDistribution!double(0.5), new LogLink!double());
  writeln(logit_link_negative_binomial_distrib);
  /* LogLink With Power Distribution */
  auto log_link_power_distrib = glm(carsX, carsY, 
      new PowerDistribution!double(0.5), new PowerLink!double(0.5));
  writeln(log_link_power_distrib);
  /* Logit Link With Binomial Distribution - Works fine */
  auto logit_link_binomial_distrib_two_col = glm(educationX, educationY, 
      new BinomialDistribution!double(), new LogitLink!double());
  writeln(logit_link_binomial_distrib_two_col);
  }
  /* Cauchit Link With Binomial Distribution */
  auto cauchit_link_binomial_distrib_two_col = glm(educationX, educationY, 
      new BinomialDistribution!double(), new CauchitLink!double());
  writeln(cauchit_link_binomial_distrib_two_col);
}

