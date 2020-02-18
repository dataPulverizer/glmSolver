/*
  This file is for the glm link functions
*/
module glmsolverd.link;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;
import glmsolverd.tools;

import std.conv: to;
import std.algorithm: min, max, fold;
import std.math: atan, exp, expm1, log, modf, fabs, fmax, fmin, cos, tan, PI;
import std.mathspecial : normalDistribution, normalDistributionInverse;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

import std.parallelism;
import std.range : iota;

alias fmin min;
alias fmax max;
alias PI pi;
alias normalDistribution pnorm;
alias normalDistributionInverse qnorm;
T dnorm(T)(T x)
{
  return (1/((2*pi)^^0.5)) * exp(-(x^^2)/2);
}
/********************************************* Accuracy ******************************************/
/* Epsilon for floats */
template eps(T)
if(isFloatingPoint!T)
{
  enum eps = T.epsilon;
}
template ceps(T)
if(isFloatingPoint!T)
{
  enum ceps = 1 - T.epsilon;
}
/******************************************* Link Functions *********************************/

/*
  It might be best to make this an abstract class so that you can have
  prototype methods with bodies without constantly repeating them later
  but you'll need to add override to all the child class methods.
*/
interface AbstractLink(T)
{
  ColumnVector!T linkfun(ColumnVector!T mu);
  T linkfun(T mu);
  ColumnVector!T deta_dmu(ColumnVector!T mu, ColumnVector!T eta);
  T deta_dmu(T mu, T eta);
  ColumnVector!T linkinv(ColumnVector!T eta);
  T linkinv(T eta);
  string toString();
  /* Block Matrix Overloads */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu);
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta);
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta);

  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu);
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta);
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta);
}

//mixin template BlockMethodGubbings(T)
//{
//  /* Block Matrix Overloads for now use for loops rather than maps */
//  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
//  {
//    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
//    ulong n = mu.length;
//    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
//    for(ulong i = 0; i < n; ++i)
//      ret[i] = linkfun(mu[i]);
//    return ret;
//  }
//  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
//  {
//    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
//    ulong n = mu.length;
//    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
//    for(ulong i = 0; i < n; ++i)
//      ret[i] = deta_dmu(mu[i], eta[i]);
//    return ret;
//  }
//  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
//  {
//    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
//    ulong n = eta.length;
//    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
//    for(ulong i = 0; i < n; ++i)
//      ret[i] = linkinv(eta[i]);
//    return ret;
//  }
//}

class LogLink(T): AbstractLink!T
{
  T linkfun(T mu)
  {
    return log(mu);
  }
  ColumnVector!T linkfun(ColumnVector!T mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return mu^^-1;
  }
  ColumnVector!T deta_dmu(ColumnVector!T mu, ColumnVector!T eta)
  {
    return map!((T m, T x) => deta_dmu(m, x))(mu, eta);
  }
  T linkinv(T eta)
  {
    return exp(eta);
  }
  ColumnVector!T linkinv(ColumnVector!T eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "LogLink";
  }
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class IdentityLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return mu;
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return mu.dup;
  }
  T deta_dmu(T mu, T eta)
  {
    return 1;
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return fillColumn!(T)(1, eta.len);
  }
  T linkinv(T eta)
  {
    return eta;
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return eta.dup;
  }
  override string toString()
  {
    return "IdentityLink";
  }
  /* Block Matrix Overloads */
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class InverseLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return mu^^-1;
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => x^^-1 )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return -(mu^^-2);
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T t) => deta_dmu(m, t) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return eta^^-1;
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T t) => t^^-1 )(eta);
  }
  override string toString()
  {
    return "InverseLink";
  }
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class LogitLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return log(mu/(1 - mu));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T m) => linkfun(m) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return (mu * (1 - mu))^^-1;
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    T x = exp(eta);
    return x/(1 + x);
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "LogitLink";
  }
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class CauchitLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return tan(pi * (mu - 0.5));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T m) => linkfun(m) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return pi * (cos(pi * (mu - 0.5))^^(-2));
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T t) => deta_dmu(m, t) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return (atan(eta)/pi) + 0.5;
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "CauchitLink";
  }
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class ProbitLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return qnorm(mu);
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!(qnorm)(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return dnorm(eta)^^-1;
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!(dnorm)(eta);
  }
  T linkinv(T eta)
  {
    return pnorm(eta);
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "ProbitLink";
  }
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class PowerLink(T) : AbstractLink!(T)
{
  immutable(T) alpha;
  LogLink!T logl;
  T linkfun(T mu)
  {
    return alpha == 0 ? logl.linkfun(mu) : mu^^alpha;
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return alpha == 0 ? logl.deta_dmu(mu, eta) : alpha * (mu^^(alpha - 1));
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return alpha == 0 ? logl.linkinv(eta) : eta^^(alpha^^-1);
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "PowerLink{" ~ to!string(alpha) ~ "}";
  }
  this(T _alpha)
  {
    alpha = _alpha;
    logl = new LogLink!T();
  }
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class OddsPowerLink(T) : AbstractLink!(T)
{
  immutable(T) alpha;
  LogitLink!(T) logit;
  T linkfun(T mu)
  {
    return alpha == 0 ? logit.linkfun(mu) : ((mu/(1 - mu))^^alpha - 1)/alpha;
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return alpha == 0 ? logit.deta_dmu(mu, eta) : (mu^^(alpha - 1))/((1 - mu)^^(alpha + 1));
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    T ret;
    if(alpha == 0)
    {
      return logit.linkinv(eta);
    }else{
      T tmp = ((eta * alpha + 1)^^(1/alpha));
      ret = min(max(tmp/(1 + tmp), eps!(T)), ceps!(T));
    }
    return ret;
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "OddsPowerLink";
  }
  this(T _alpha)
  {
    alpha = _alpha;
    logit = new LogitLink!(T)();
  }
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class LogComplementLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return log(max(1 - mu, eps!(T)));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return -(max(1 - mu, eps!(T)))^^(-1);
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return min(max(-expm1(eta), eps!(T)), ceps!(T));
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "LogComplementLink";
  }
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class LogLogLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return -log(-log(mu));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return -(mu * log(mu))^^-1;
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return exp(-exp(-eta));
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "LogLogLink";
  }
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class ComplementaryLogLogLink(T) : AbstractLink!(T)
{
  T linkfun(T mu)
  {
    return log(-log(1 - mu));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return ((mu - 1) * log(1 - mu))^^-1;
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    return 1 - exp(-exp(eta));
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "LogLogLink";
  }
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}
class NegativeBinomialLink(T) : AbstractLink!(T)
{
  immutable(T) alpha;
  T linkfun(T mu)
  {
    T ialpha = alpha^^-1;
    return log(mu/(mu + ialpha));
  }
  ColumnVector!(T) linkfun(ColumnVector!(T) mu)
  {
    return map!( (T x) => linkfun(x) )(mu);
  }
  T deta_dmu(T mu, T eta)
  {
    return 1/(mu + alpha * mu^^2);
  }
  ColumnVector!(T) deta_dmu(ColumnVector!(T) mu, ColumnVector!(T) eta)
  {
    return map!( (T m, T x) => deta_dmu(m, x) )(mu, eta);
  }
  T linkinv(T eta)
  {
    T tmp = exp(eta);
    return tmp/(alpha * (1 - tmp));
  }
  ColumnVector!(T) linkinv(ColumnVector!(T) eta)
  {
    return map!( (T x) => linkinv(x) )(eta);
  }
  override string toString()
  {
    return "NegativeBinomialLink{"~ to!string(alpha) ~ "}";
  }
  this(T _alpha)
  {
    alpha = _alpha;
  }
  //mixin BlockMethodGubbings!(T);
  /* Block Matrix Overloads for now use for loops rather than maps */
  BlockColumnVector!(T) linkfun(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => linkfun(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    //return map!((ColumnVector!(T) m, ColumnVector!(T) x) => deta_dmu(m, x))(mu, eta);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(BlockColumnVector!(T) eta)
  {
    //return map!( (ColumnVector!(T) x) => linkinv(x) )(eta);
    ulong n = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = linkinv(eta[i]);
    return ret;
  }

  /* For parallel block algorithms */
  BlockColumnVector!(T) linkfun(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkfun(mu[i]);
    return ret;
  }
  BlockColumnVector!(T) deta_dmu(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = deta_dmu(mu[i], eta[i]);
    return ret;
  }
  BlockColumnVector!(T) linkinv(Block1DParallel dataType, BlockColumnVector!(T) eta)
  {
    ulong nBlocks = eta.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = linkinv(eta[i]);
    return ret;
  }
}

