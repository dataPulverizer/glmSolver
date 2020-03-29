module glmsolverd.sample;

import std.conv: to;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;
import glmsolverd.link;
import glmsolverd.distributions;
import glmsolverd.tools;
import glmsolverd.linearalgebra;

import std.stdio : writeln;
import std.typecons: Tuple, tuple;
import std.random: Mt19937_64, uniform;

import std.math: abs, exp, fmax, pow;
alias fmax max;

import std.algorithm.iteration: mean;


Matrix!(T) randomCorrelationMatrix(T)(ulong d, ulong seed)
{
  auto r = createRandomMatrix!(T)(d, seed);
  //writeln("Initial random matrix: ", r);
  r = cast(T)(0.5) * (r + r.t());
  for(ulong i = 0; i < d; ++i)
    r[i, i] = cast(T)(1);
  
  auto ev = eigen(r);
  //auto ev = eigenGeneral(r);
  //auto ev = eigenGeneralX(r);
  
  //const CBLAS_LAYOUT layout = CblasColMajor;
  
  auto eigenvec = new Matrix!(T, CblasColMajor)(ev.vectors, [d, d]);
  //writeln("Eigen values", ev.values);
  //writeln("Eigen vectors", new Matrix!(T, CblasColMajor)(ev.vectors, [d, d]));
  auto vec1 = eigenvec.dup;
  
  for(ulong j = 0; j < d; ++j)
    for(ulong i = 0; i < d; ++i)
      vec1[i, j] *= abs(ev.values[j]);
  r = mult_!(T, CblasColMajor, CblasNoTrans, CblasTrans)(vec1, eigenvec);
  
  //for(ulong i = 0; i < d; ++i)
  //  ev.values[i] = abs(ev.values[i]);
  //r = mult_!(T, CblasColMajor, CblasNoTrans, CblasNoTrans)(mult_!(T, CblasColMajor)(eigenvec, diag(ev.values)), eigenvec.t());
  
  T _max = abs(r[0, 0]);
  for(ulong j = 0; j < d; ++j)
    for(ulong i = 0; i < d; ++i)
    {
      T _abs = abs(r[i, j]);
      if(_max < _abs)
        _max = _abs;
    }
  
  //writeln("Maximum R: ", _max);
  
  r /= _max;
  r = cast(T)(0.5) * (r + r.t());
  for(ulong i = 0; i < d; ++i)
    r[i, i] = cast(T)(1);
  
  return r;
}

/*
  Simulate from multivariate normal distribution
*/
import std.mathspecial: normalDistributionInverse;
Matrix!(T, layout) mvrnorm(T, CBLAS_LAYOUT layout = CblasColMajor)
  (ulong n, ColumnVector!(T) mu, Matrix!(T, layout) sigma, ulong seed)
{
  ulong p = sigma.ncol;
  auto A = cholesky!('L', T, layout)(sigma);
  auto output = fillMatrix!(T, layout)(cast(T)(0), [n, p]);
  
  for(ulong i = 0; i < n; ++i)
  {
    auto rndVec = sampleStandardNormal!(T)(p, ++seed);
    auto tmp = mu + mult_(A, rndVec);
    for(ulong j = 0; j < p; ++j)
      output[i, j] = tmp[j];
  }

  return output;
}

/*
  Function to simulate X and eta, p = number of parameters,
  n = number of samples.
*/
auto simulateData(T, CBLAS_LAYOUT layout = CblasColMajor)
  (ulong p, ulong n, ulong seed, T delta = cast(T)(0))
{
  auto corr = randomCorrelationMatrix!(T)(p, seed);
  auto mu = zerosColumn!(T)(p);
  auto X = mvrnorm!(T, layout)(n, mu, corr, ++seed);

  /* The intercept */
  for(ulong i = 0; i < n; ++i)
    X[i, 0] = cast(T)(1);

  auto b = zerosColumn!(T)(p);
  Mt19937_64 rng;
  rng.seed(seed);

  /* The intercept */
  b[0] = uniform!("()")(cast(T)(0), cast(T)(0.3), rng);

  if(b.length > 1)
  {
    for(ulong i = 1; i < p; ++i)
      b[i] = uniform!("()")(cast(T)(-0.1), cast(T)(0.1), rng);
  }

  auto eta = mult_(X, b);
  T sd = 0.5 * abs(mean(eta.getData));
  eta += delta + (sd * sampleStandardNormal!(T)(n, ++seed));

  return tuple!("X", "eta")(X, eta);
}

/*
  Function to sample a vector from  Poisson Distribution
  when given a vector of lambdas
*/
ColumnVector!(T) _sample_poisson(T)(ColumnVector!(T) lambda, ulong seed)
{
  ulong n = lambda.length;
  auto vec = zerosColumn!(T)(n);
  
  Mt19937_64 rng;
  rng.seed(seed);
  
  for(ulong i = 0; i < n; ++i)
  {
    T p = exp(-lambda[i]);
    T F = p; T x = 0;
    T u = uniform!("()")(cast(T)(0), cast(T)(1), rng);
    while(true)
    {
      if(u < F)
        break;
      x += 1;
      p *= lambda[i]/x;
      F += p;
    }
    vec[i] = x;
  }
  
  return vec;
}

/*
  Function to simulate data for GLM
*/
auto simulateData(T, CBLAS_LAYOUT layout = CblasColMajor)
  (AbstractDistribution!(T) distrib, AbstractLink!(T) link,
  ulong p, ulong n, ulong seed)
{
  auto Xy = simulateData!(T, layout)(p, n, seed);
  auto _y = link.linkinv(Xy.eta);
  
  if(distrib.toString() == "PoissonDistribution")
  {
    _y = _sample_poisson(_y, ++seed);
    //writeln("Length: ", _y.getData);
  }
  
  if(distrib.toString() == "BinomialDistribution")
  {
    Mt19937_64 rng;
    rng.seed(++seed);
    /* Add extra noise to the data */
    _y = map!((x) => cast(T)(1) * (x > uniform!("()")(cast(T)(0), cast(T)(1), rng)))(_y);
    //writeln("Length: ", _y.getData);
  }

  if(distrib.toString() == "GammaDistribution")
  {
    _y += 10;
  }
  
  auto y = new Matrix!(T, layout)(_y.getData, [n, cast(ulong)1]);

  return tuple!("X", "y")(Xy.X, y);
}



