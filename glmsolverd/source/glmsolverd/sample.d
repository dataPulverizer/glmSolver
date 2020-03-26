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
import std.random: Random, uniform;

import std.math: abs, pow, fmax;
alias fmax max;

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

