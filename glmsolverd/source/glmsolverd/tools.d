/*
  Error functions, control object and GLM Object
*/
module glmsolverd.tools;

import glmsolverd.arrays;
import glmsolverd.apply;
import glmsolverd.link;
import glmsolverd.distributions;

import std.math: fabs;
import std.conv: to;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

import std.parallelism;
import std.range : iota;

/******************************************* Weight & Systematic Component *********************************/
auto Z(T)(AbstractLink!T link, T y, T mu, T eta)
{
  return link.deta_dmu(mu, eta) * (y - mu) + eta;
}
auto Z(T)(AbstractLink!T link, ColumnVector!T y, 
          ColumnVector!T mu, ColumnVector!T eta)
{
  return map!( (T x, T m, T t) => link.Z(x, m, t) )(y, mu, eta);
}
auto Z(T)(AbstractLink!T link, BlockColumnVector!T y, 
          BlockColumnVector!T mu, BlockColumnVector!T eta)
{
  ulong n = y.length;
  BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
  for(ulong i = 0; i < n; ++i)
    ret[i] = Z(link, y[i], mu[i], eta[i]);
  return ret;
}
auto Z(T)(Block1DParallel dataType, AbstractLink!T link, BlockColumnVector!T y, 
          BlockColumnVector!T mu, BlockColumnVector!T eta)
{
  ulong nBlocks = y.length;
  BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
  foreach(i; taskPool.parallel(iota(nBlocks)))
    ret[i] = Z(link, y[i], mu[i], eta[i]);
  return ret;
}


/* Weights Vector */
auto W(T)(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
{
  return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^-1;
}
auto W(T)(AbstractDistribution!T distrib, AbstractLink!T link, 
          ColumnVector!T mu, ColumnVector!T eta)
{
  return map!( (T m, T x) => W!(T)(distrib, link, m, x) )(mu, eta);
}
/* Square Root of Weights Vector */
auto WS(T)(AbstractDistribution!T distrib, AbstractLink!T link, T mu, T eta)
{
  return ((link.deta_dmu(mu, eta)^^2) * distrib.variance(mu))^^-0.5;
}
auto WS(T)(AbstractDistribution!T distrib, AbstractLink!T link,
          ColumnVector!T mu, ColumnVector!T eta)
{
  return map!( (T m, T x) => WS!(T)(distrib, link, m, x) )(mu, eta);
}
/******************************************* Error Functions *********************************/
alias fabs abs;

T absoluteError(T)(T x, T y)
if(isFloatingPoint!T)
{
  return abs(x - y);
}
T absoluteError(T)(Vector!T x, Vector!T y)
if(isFloatingPoint!T)
{
  assert(x.len == y.len, "vectors are not equal.");
  T ret = 0;
  for(ulong i = 0; i < x.len; ++i)
    ret += (x[i] - y[i])^^2;
  return (ret^^0.5);
}
T relativeError(T)(T x, T y)
if(isFloatingPoint!T)
{
  return abs(x - y)/(0.1 + abs(x));
}
T relativeError(T)(Vector!T x, Vector!T y)
if(isFloatingPoint!T)
{
  assert(x.len == y.len, "vectors are not equal.");
  T x1 = 0; T x2 = 0;
  for(ulong i = 0; i < x.len; ++i)
  {
    x1 += (x[i] - y[i])^^2;
    x2 += x[i]^^2;
  }
  return (x1^^0.5)/(1E-5 + (x2^^0.5));
}
/****************************************** Control Class *****************************************/
class Control(T)
if(isFloatingPoint!T)
{
  immutable(T) epsilon;
  immutable(int) maxit;
  immutable(bool) printError;
  immutable(bool) printCoef;
  immutable(T) minstep;
  this(int _maxit = 25, T _epsilon = 1E-7,
      bool _printError = false, bool _printCoef = false, 
      T _minstep = 1E-5)
  {
    epsilon = _epsilon; maxit = _maxit;
    printError = _printError; printCoef = _printCoef;
    minstep = _minstep;
  }
  override string toString() const
  {
    string repr = "Control(T = " ~ T.stringof ~ ")\n{\n";
    repr ~= "  Epsilon = " ~ to!string(epsilon) ~ ";\n";
    repr ~= "  Maxit = " ~ to!string(maxit) ~ ";\n";
    repr ~= "  Print Error = " ~ to!string(printError) ~ ";\n";
    repr ~= "  Minstep = " ~ to!string(printCoef) ~ ";\n";
    repr ~= "}\n";
    return repr;
  }
}
/**************************************** GLM Result Class ***************************************/
class GLM(T, CBLAS_LAYOUT L)
if(isFloatingPoint!T)
{
  ulong niter;
  bool converged;
  AbstractDistribution!T distrib;
  AbstractLink!T link;
  T phi;
  T[] coefficients;
  T[] standardError;
  Matrix!(T, L) cov;
  T deviance;
  T absoluteError;
  T relativeError;
  this(T, CBLAS_LAYOUT L)(ulong _niter, bool _converged, T _phi, AbstractDistribution!T _distrib, AbstractLink!T _link,
      ColumnVector!T coeff, Matrix!(T, L) _cov, T _deviance, T absErr, T relErr)
  {
    niter = _niter;
    converged = _converged;
    distrib = _distrib;
    link = _link;
    phi = _phi;
    coefficients = coeff.getData;
    standardError = new T[_cov.nrow];
    for(ulong i = 0; i < _cov.nrow; ++i)
      standardError[i] = _cov[i, i]^^0.5;
    cov = _cov;
    deviance = _deviance;
    absoluteError = absErr;
    relativeError = relErr;
  }
  override string toString()
  {
    string rep = "GLM(" ~ link.toString() ~ ", " ~ distrib.toString() ~ ")\n";
    rep ~= "Info(Converged = " ~ to!string(converged) ~ ", Iterations = " ~ to!string(niter) ~ ")\n";
    rep ~= "Error(Absolute Error = " ~ to!string(absoluteError) ~
            ", Relative Error = " ~ to!string(relativeError) ~ 
            ", Deviance = " ~ to!string(deviance) ~ ", phi = " ~ 
            to!string(phi) ~ ")\n";
    rep ~= "Coefficients:\n" ~ to!string(coefficients) ~ "\n";
    rep ~= "StandardError:\n" ~ to!string(standardError) ~ "\n";
    return rep;
  }
}

interface AbstractMatrixType {}
class RegularData : AbstractMatrixType {}
class Block1D : AbstractMatrixType {}
class Block1DParallel : AbstractMatrixType {}
