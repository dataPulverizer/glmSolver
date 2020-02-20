/*
  Module for distributions for GLM
*/
module glmsolverd.distributions;

import glmsolverd.apply;
import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.tools;

import std.conv: to;
import std.math: log;
import std.typecons: Tuple, tuple;

import std.parallelism;
import std.range : iota;

/******************************************* Distributions *********************************/
template initType(T)
{
  alias initType = Tuple!(T, T, T);
}
/* Probability Distributions */
abstract class AbstractDistribution(T)
{
  initType!(ColumnVector!T) init(Matrix!(T) _y, ColumnVector!T wts)
  {
    auto y = cast(ColumnVector!(T))_y;
    return tuple(y, y.dup, wts);
  }
  
  ColumnVector!T variance(ColumnVector!T mu);
  T variance(T mu);
  ColumnVector!T devianceResiduals(ColumnVector!T mu, ColumnVector!T y);
  T devianceResiduals(T mu, T y);
  T devianceResiduals(T mu, T y, T wts);
  ColumnVector!T devianceResiduals(ColumnVector!T mu, ColumnVector!T y, ColumnVector!T wts);
  
  /* Block Matrix/Vector Overloads */
  initType!(BlockColumnVector!(T)) init(BlockMatrix!(T) _y, BlockColumnVector!(T) wts)
  {
    //BlockColumnVector!(T) y = map!( (Matrix!(T) x) => cast(ColumnVector!(T))x )(_y);
    //BlockColumnVector!(T) mu = map!( (ColumnVector!(T) x) => x.dup )(_y);
    ulong n = _y.length;
    BlockColumnVector!(T) y = new ColumnVector!(T)[n];
    BlockColumnVector!(T) mu = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
    {
      y[i] = cast(ColumnVector!(T))_y[i];
      mu[i] = (cast(ColumnVector!(T))_y[i]).dup;
    }
    return tuple(y, mu, wts);
  }
  initType!(BlockColumnVector!(T)) init(Block1DParallel dataType, BlockMatrix!(T) _y, BlockColumnVector!(T) wts)
  {
    ulong nBlocks = _y.length;
    BlockColumnVector!(T) y = new ColumnVector!(T)[nBlocks];
    BlockColumnVector!(T) mu = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      y[i] = cast(ColumnVector!(T))_y[i];
      mu[i] = (cast(ColumnVector!(T))_y[i]).dup;
    }
    return tuple(y, mu, wts);
  }
  BlockColumnVector!(T) variance(BlockColumnVector!(T) mu);
  
  //BlockColumnVector!(T) variance(Block1DParallel dataType, BlockColumnVector!(T) mu);
  BlockColumnVector!(T) variance(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = variance(mu[i]);
    return ret;
  }
  
  BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y);
  BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts);

  BlockColumnVector!(T) devianceResiduals(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) y)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = devianceResiduals(mu[i], y[i]);
    return ret;
  }
  BlockColumnVector!(T) devianceResiduals(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = devianceResiduals(mu[i], y[i], wts[i]);
    return ret;
  }
}
T y_log_y(T)(T y, T x)
{
  //pragma(inline, true);
  return y != 0 ? y * log(y/x) : 0;
}
template BlockDistributionGubbings()
{
  enum string BlockDistributionGubbings = q{
  override BlockColumnVector!(T) variance(BlockColumnVector!(T) mu)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = variance(mu[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i], wts[i]);
    return ret;
  }
};
}
class BinomialDistribution(T) : AbstractDistribution!(T)
{
  override initType!(ColumnVector!T) init(Matrix!T _y, ColumnVector!T wts)
  {
    ColumnVector!(T) y; ColumnVector!T mu;
    bool hasWeights = wts.len > 0;
    if(_y.ncol == 1)
    {
      y = cast(ColumnVector!(T))_y;
      if(wts.len == 0)
      {
        mu = map!( (T x) => (x + cast(T)0.5)/2 )(y);
      }else{
        mu = map!( (T x, T w) => (w * x + cast(T)0.5)/(w + 1) )(y, wts);
      }
    }else if(_y.ncol > 1)
    {
      y = new ColumnVector!(T)(_y.nrow);
      mu = new ColumnVector!(T)(_y.nrow);
      wts = new ColumnVector!(T)(_y.nrow);
      for(ulong i = 0; i < _y.nrow; ++i)
      {
        wts[i] = _y[i, 0] + _y[i, 1];
        y[i] = _y[i, 0]/wts[i];
        mu[i] = (wts[i] * y[i] + 0.5)/(wts[i] + 1);
      }
    }
    return tuple(y, mu, wts);
  }
  override T variance(T mu)
  {
    return mu * (1 - mu);
  }
  override ColumnVector!T variance(ColumnVector!T mu)
  {
    return map!( (T x) => variance(x) )(mu);
  }
  override T devianceResiduals(T mu, T y)
  {
    return 2*(y_log_y!(T)(y, mu) + y_log_y!(T)(1 - y, 1 - mu));
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!T mu, ColumnVector!T y)
  {
    return map!((T m, T x) => devianceResiduals(m, x))(mu, y);
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    return 2 * wts * (y_log_y!(T)(y, mu) + y_log_y!(T)(1 - y, 1 - mu));
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!T mu, ColumnVector!T y, ColumnVector!T wts)
  {
    return map!( (T m, T x, T w) => devianceResiduals(m, x, w) )(mu, y, wts);
  }
  override string toString()
  {
    return "BinomialDistribution";
  }
  override initType!(BlockColumnVector!T) init(BlockMatrix!T _y, BlockColumnVector!T wts)
  {
    BlockColumnVector!(T) y; BlockColumnVector!T mu;
    bool hasWeights = wts.length > 0;
    if(_y[0].ncol == 1)
    {
      //y = map!( (Matrix!(T) x) => cast(ColumnVector!(T))x )(y);
      y = new ColumnVector!(T)[_y.length];
      for(ulong i = 0; i < y.length; ++i)
        y[i] = cast(ColumnVector!(T))_y[i];
      if(wts.length == 0)
      {
        //mu = map!( (T x) => (x + cast(T)0.5)/2 )(y);
        //mu = map!((ColumnVector!(T) m) => map!( (T x) => (x + cast(T)0.5)/2 )(m))(y);
        mu = new ColumnVector!(T)[y.length];
        for(ulong i = 0; i < y.length; ++i)
        {
          mu[i] = y[i].dup; ulong b = y[i].length;
          for(ulong j = 0; j < b; ++j)
          {
            mu[i][j] = (mu[i][j] + cast(T)0.5)/2;
          }
        }
      }else{
        //mu = map!( (T x, T w) => (w * x + cast(T)0.5)/(w + 1) )(y, wts);
        //mu = map!((ColumnVector!(T) m, ColumnVector!(T) w) => map!( (T x, T ws) => (ws * x + cast(T)0.5)/(ws + 1) )(m, w))(y, wts);
        y = new ColumnVector!(T)[_y.length];
        mu = new ColumnVector!(T)[_y.length];
        //wts = new ColumnVector!(T)[_y.length];
        for(ulong i = 0; i < _y.length; ++i)
        {
          ulong b = _y[i].ncol;
          y[i] = cast(ColumnVector!(T))_y[i];
          mu[i] = y[i].dup;
          for(ulong j = 0; j < b; ++j)
          {
            mu[i][j] = (wts[i][j] * mu[i][j] + cast(T)0.5)/(wts[i][j] + 1);
          }
        }
      }
    }else if(_y[0].ncol > 1)
    {
      ulong n = _y.length;
      y = new ColumnVector!(T)[n];
      mu = new ColumnVector!(T)[n];
      wts = new ColumnVector!(T)[n];
      for(ulong i = 0; i < n; ++i)
      {
        ulong m = _y[i].nrow;
        y[i] = zerosColumn!(T)(m);
        mu[i] = zerosColumn!(T)(m);
        wts[i] = zerosColumn!(T)(m);
        for(ulong j = 0; j < m; ++j)
        {
          wts[i][j] = _y[i][j, 0] + _y[i][j, 1];
          y[i][j] = _y[i][j, 0]/wts[i][j];
          mu[i][j] = (wts[i][j] * y[i][j] + 0.5)/(wts[i][j] + 1);
        }
      }
    }
    return tuple(y, mu, wts);
  }
  override initType!(BlockColumnVector!T) init(Block1DParallel dataType, BlockMatrix!T _y, BlockColumnVector!T wts)
  {
    BlockColumnVector!(T) y; BlockColumnVector!T mu;
    bool hasWeights = wts.length > 0;
    immutable(ulong) nBlocks = _y.length;
    if(_y[0].ncol == 1)
    {
      y = new ColumnVector!(T)[nBlocks];

      foreach(i; taskPool.parallel(iota(nBlocks)))
        y[i] = cast(ColumnVector!(T))_y[i];
      
      if(wts.length == 0)
      {
        mu = new ColumnVector!(T)[nBlocks];
        foreach(i; taskPool.parallel(iota(nBlocks)))
        {
          mu[i] = y[i].dup; ulong b = y[i].length;
          for(ulong j = 0; j < b; ++j)
          {
            mu[i][j] = (mu[i][j] + cast(T)0.5)/2;
          }
        }
      }else{
        y = new ColumnVector!(T)[nBlocks];
        mu = new ColumnVector!(T)[nBlocks];

        foreach(i; taskPool.parallel(iota(nBlocks)))
        {
          ulong b = _y[i].ncol;
          y[i] = cast(ColumnVector!(T))_y[i];
          mu[i] = y[i].dup;
          for(ulong j = 0; j < b; ++j)
          {
            mu[i][j] = (wts[i][j] * mu[i][j] + cast(T)0.5)/(wts[i][j] + 1);
          }
        }
      }
    }else if(_y[0].ncol > 1)
    {
      y = new ColumnVector!(T)[nBlocks];
      mu = new ColumnVector!(T)[nBlocks];
      wts = new ColumnVector!(T)[nBlocks];

      foreach(i; taskPool.parallel(iota(nBlocks)))
      {
        ulong m = _y[i].nrow;
        y[i] = zerosColumn!(T)(m);
        mu[i] = zerosColumn!(T)(m);
        wts[i] = zerosColumn!(T)(m);
        for(ulong j = 0; j < m; ++j)
        {
          wts[i][j] = _y[i][j, 0] + _y[i][j, 1];
          y[i][j] = _y[i][j, 0]/wts[i][j];
          mu[i][j] = (wts[i][j] * y[i][j] + 0.5)/(wts[i][j] + 1);
        }
      }
    }
    return tuple(y, mu, wts);
  }
  //mixin BlockDistributionGubbings!();
  override BlockColumnVector!(T) variance(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => variance(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = variance(mu[i]);
    return ret;
  }
  override BlockColumnVector!(T) variance(Block1DParallel dataType, BlockColumnVector!(T) mu)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = variance(mu[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2) => devianceResiduals(x1, x2) )(mu, y);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2, ColumnVector!(T) x3) => devianceResiduals(x1, x2, x3) )(mu, y, wts);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i], wts[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) y)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = devianceResiduals(mu[i], y[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(Block1DParallel dataType, BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts)
  {
    ulong nBlocks = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      ret[i] = devianceResiduals(mu[i], y[i], wts[i]);
    return ret;
  }
}
class PoissonDistribution(T) : AbstractDistribution!(T)
{
  override initType!(ColumnVector!T) init(Matrix!(T) _y, ColumnVector!T wts)
  {
    auto y = cast(ColumnVector!(T))_y;
    ColumnVector!(T) mu = map!( (T x) => x + 0.1 )(y);
    return tuple(y, mu, wts);
  }
  override T variance(T mu)
  {
    return mu;
  }
  override ColumnVector!(T) variance(ColumnVector!(T) mu)
  {
    return mu.dup;
  }
  override T devianceResiduals(T mu, T y)
  {
    T dev;
    if(y == 0)
      dev = 2 * mu;
    else if(y > 0)
      dev = 2 * (y * log(y/mu) - (y - mu));
    return dev;
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y)
  {
    return map!((T m, T x) => devianceResiduals(m, x))(mu, y);
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    T dev;
    if(y == 0)
      dev = 2 * wts * mu;
    else if(y > 0)
      dev = 2 * wts * (y * log(y/mu) - (y - mu));
    return dev;
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y, ColumnVector!(T) wts)
  {
    return map!((T m, T x, T w) => devianceResiduals(m, x, w))(mu, y, wts);
  }
  override string toString()
  {
    return "PoissonDistribution";
  }
  /* Block Matrix/Vector Overloads */
  override initType!(BlockColumnVector!(T)) init(BlockMatrix!(T) _y, BlockColumnVector!(T) wts)
  {
    ulong n = _y.length;
    BlockColumnVector!(T) y = new ColumnVector!(T)[n];
    BlockColumnVector!(T) mu = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      y[i] = cast(ColumnVector!(T))_y[i];
    for(ulong i = 0; i < n; ++i)
      mu[i] = map!( (T m) => m + 0.1 )(y[i]);
    return tuple(y, mu, wts);
  }
  override initType!(BlockColumnVector!(T)) init(Block1DParallel dataType, BlockMatrix!(T) _y, BlockColumnVector!(T) wts)
  {
    ulong nBlocks = _y.length;
    BlockColumnVector!(T) y = new ColumnVector!(T)[nBlocks];
    BlockColumnVector!(T) mu = new ColumnVector!(T)[nBlocks];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      y[i] = cast(ColumnVector!(T))_y[i];
    foreach(i; taskPool.parallel(iota(nBlocks)))
      mu[i] = map!( (T m) => m + 0.1 )(y[i]);
    return tuple(y, mu, wts);
  }
  //mixin BlockDistributionGubbings!();
  override BlockColumnVector!(T) variance(BlockColumnVector!(T) mu)
  {
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = variance(mu[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2) => devianceResiduals(x1, x2) )(mu, y);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2, ColumnVector!(T) x3) => devianceResiduals(x1, x2, x3) )(mu, y, wts);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i], wts[i]);
    return ret;
  }
}
class GaussianDistribution(T) : AbstractDistribution!(T)
{
  override T variance(T mu)
  {
    return cast(T)1;
  }
  override ColumnVector!(T) variance(ColumnVector!(T) mu)
  {
    return onesColumn!T(mu.len);
  }
  override T devianceResiduals(T mu, T y)
  {
    return (y - mu)^^2;
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    return wts * (y - mu)^^2;
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y)
  {
    return map!( (T m, T x) => devianceResiduals(m, x) )(mu, y);
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y, ColumnVector!(T) wts)
  {
    return map!( (T m, T x, T w) => devianceResiduals(m, x, w) )(mu, y, wts);
  }
  override string toString()
  {
    return "GaussianDistribution";
  }
  //mixin BlockDistributionGubbings!();
  override BlockColumnVector!(T) variance(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => variance(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = variance(mu[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2) => devianceResiduals(x1, x2) )(mu, y);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2, ColumnVector!(T) x3) => devianceResiduals(x1, x2, x3) )(mu, y, wts);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i], wts[i]);
    return ret;
  }
}
class InverseGaussianDistribution(T) : AbstractDistribution!(T)
{
  override T variance(T mu)
  {
    return mu^^3;
  }
  override ColumnVector!(T) variance(ColumnVector!(T) mu)
  {
    return map!( (T m) => m^^3 )(mu);
  }
  override T devianceResiduals(T mu, T y)
  {
    return ((y - mu)^^2)/(y * (mu^^2));
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    return wts * ((y - mu)^^2)/(y * (mu^^2));
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y)
  {
    return map!( (T m, T x) => devianceResiduals(m, x) )(mu, y);
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y, ColumnVector!(T) wts)
  {
    return map!( (T m, T x, T w) => devianceResiduals(m, x, w) )(mu, y, wts);
  }
  override string toString()
  {
    return "InverseGaussianDistribution";
  }
  //mixin BlockDistributionGubbings!();
  override BlockColumnVector!(T) variance(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => variance(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = variance(mu[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2) => devianceResiduals(x1, x2) )(mu, y);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2, ColumnVector!(T) x3) => devianceResiduals(x1, x2, x3) )(mu, y, wts);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i], wts[i]);
    return ret;
  }
}
class NegativeBinomialDistribution(T) : AbstractDistribution!(T)
{
  immutable(T) alpha; /* Parameter */
  override initType!(ColumnVector!(T)) init(Matrix!(T) _y, ColumnVector!(T) wts)
  {
    auto y = cast(ColumnVector!(T))_y;
    T tmp = 1/6;
    auto mu = map!( (T x) => x == 0 ? tmp : x)(y);
    return tuple(y, mu, wts);
  }
  override initType!(BlockColumnVector!(T)) init(BlockMatrix!(T) _y, BlockColumnVector!(T) wts)
  {
    ulong n = _y.length;
    BlockColumnVector!(T) y = new ColumnVector!(T)[n];
    BlockColumnVector!(T) mu = new ColumnVector!(T)[n];
    T tmp = 1/6;
    for(ulong i = 0; i < n; ++i)
      y[i] = cast(ColumnVector!(T))_y[i];
    for(ulong i = 0; i < n; ++i)
      mu[i] = map!( (T x) => x == 0 ? tmp : x)(y[i]);
    return tuple(y, mu, wts);
  }
  override initType!(BlockColumnVector!(T)) init(Block1DParallel dataType, BlockMatrix!(T) _y, BlockColumnVector!(T) wts)
  {
    ulong nBlocks = _y.length;
    BlockColumnVector!(T) y = new ColumnVector!(T)[nBlocks];
    BlockColumnVector!(T) mu = new ColumnVector!(T)[nBlocks];
    //T tmp = 1/6;
    auto tmp = taskPool.workerLocalStorage!(double)(1/6);
    foreach(i; taskPool.parallel(iota(nBlocks)))
    {
      y[i] = cast(ColumnVector!(T))_y[i];
      mu[i] = map!( (T x) => x == 0 ? tmp.get : x)(y[i]);
    }
    return tuple(y, mu, wts);
  }
  override T variance(T mu)
  {
    return mu + alpha * (mu^^2);
  }
  override ColumnVector!(T) variance(ColumnVector!(T) mu)
  {
    return map!( (T m) => variance(m) )(mu);
  }
  override T devianceResiduals(T mu, T y)
  {
    T dev;
    T ialpha = alpha^^-1;
    if(y == 0)
      dev = 2 * ialpha * log(1/(1 + alpha*mu));
    else if(y > 0)
      dev = 2 * (y * log(y/mu) - (y + ialpha) * log((1 + alpha * y)/(1 + alpha * mu)));
    return dev;
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    T dev;
    T ialpha = alpha^^-1;
    if(y == 0)
      dev = 2 * wts * ialpha * log(1/(1 + alpha * mu));
    else if(y > 0)
      dev = 2 * wts * (y * log(y/mu) - (y + ialpha) * log((1 + alpha * y)/(1 + alpha * mu)));
    return dev;
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y)
  {
    return map!( (T m, T x) => devianceResiduals(m, x) )(mu, y);
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y, ColumnVector!(T) wts)
  {
    return map!( (T m, T x, T w) => devianceResiduals(m, x, w) )(mu, y, wts);
  }
  override string toString()
  {
    return "NegativeBinomialDistribution{alpha = " ~ to!string(alpha) ~ "}";
  }
  this(T _alpha)
  {
    alpha = _alpha;
  }
  //mixin BlockDistributionGubbings!();
  override BlockColumnVector!(T) variance(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => variance(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = variance(mu[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2) => devianceResiduals(x1, x2) )(mu, y);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2, ColumnVector!(T) x3) => devianceResiduals(x1, x2, x3) )(mu, y, wts);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i], wts[i]);
    return ret;
  }
}
/*
  See p 149 & 433 Of Generalized Linear Models & Extensions, 
  by J. W. Hardin & J. M. Hilbe.
*/
class PowerDistribution(T) : AbstractDistribution!(T)
{
  T k;
  override T variance(T mu)
  {
    return mu^^k;
  }
  override ColumnVector!(T) variance(ColumnVector!(T) mu)
  {
    return map!( (T m) => variance(m) )(mu);
  }
  override T devianceResiduals(T mu, T y)
  {
    T ok = 1 - k;
    T tk = 2 - k;
    return ( (2 * y/( ok * ((y^^(ok)) - (mu^^ok)) )) - (2/( tk * ((y^^(tk)) - (mu^^tk)) )) );
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    T ok = 1 - k;
    T tk = 2 - k;
    return wts * ( (2 * y/( ok * ((y^^(ok)) - (mu^^ok)) )) - (2/( tk * ((y^^(tk)) - (mu^^tk)) )) );
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y)
  {
    return map!( (T m, T x) => devianceResiduals(m, x) )(mu, y);
  }
  override ColumnVector!(T) devianceResiduals(ColumnVector!(T) mu, ColumnVector!(T) y, ColumnVector!(T) wts)
  {
    return map!( (T m, T x, T w) => devianceResiduals(m, x, w) )(mu, y, wts);
  }
  override string toString()
  {
    return "PowerDistribution{" ~ to!string(k) ~ "}";
  }
  this(T _k)
  {
    k = _k;
  }
  //mixin BlockDistributionGubbings!();
  override BlockColumnVector!(T) variance(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => variance(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = variance(mu[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2) => devianceResiduals(x1, x2) )(mu, y);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2, ColumnVector!(T) x3) => devianceResiduals(x1, x2, x3) )(mu, y, wts);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i], wts[i]);
    return ret;
  }
}
class GammaDistribution(T) : AbstractDistribution!(T)
{
  override ColumnVector!T variance(ColumnVector!T mu)
  {
    return map!((T x) => x^^2)(mu);
  }
  override T variance(T mu)
  {
    return mu^^2;
  }
  override ColumnVector!T devianceResiduals(ColumnVector!T mu, ColumnVector!T y)
  {
    return map!((T m1, T y1) => 2*(((y1 - m1)/m1) - log(y1/m1)) )(mu, y);
  }
  override T devianceResiduals(T mu, T y)
  {
    return 2*( ((y - mu)/mu) - log(y/mu) );
  }
  override T devianceResiduals(T mu, T y, T wts)
  {
    return 2*wts*( ((y - mu)/mu) - log(y/mu) );
  }
  override ColumnVector!T devianceResiduals(ColumnVector!T mu, ColumnVector!T y, ColumnVector!T wts)
  {
    return map!((T m1, T y1, T wts1) => 2*wts1*(((y1 - m1)/m1) - log(y1/m1)) )(mu, y, wts);
  }
  override string toString()
  {
    return "GammaDistribution";
  }
  //mixin BlockDistributionGubbings!();
  override BlockColumnVector!(T) variance(BlockColumnVector!(T) mu)
  {
    //return map!( (ColumnVector!(T) x) => variance(x) )(mu);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = variance(mu[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2) => devianceResiduals(x1, x2) )(mu, y);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i]);
    return ret;
  }
  override BlockColumnVector!(T) devianceResiduals(BlockColumnVector!(T) mu, BlockColumnVector!(T) y, BlockColumnVector!(T) wts)
  {
    //return map!( (ColumnVector!(T) x1, ColumnVector!(T) x2, ColumnVector!(T) x3) => devianceResiduals(x1, x2, x3) )(mu, y, wts);
    ulong n = mu.length;
    BlockColumnVector!(T) ret = new ColumnVector!(T)[n];
    for(ulong i = 0; i < n; ++i)
      ret[i] = devianceResiduals(mu[i], y[i], wts[i]);
    return ret;
  }
}

/* Evaluates if the dispersion should be 1 */
template unitDispsersion(T, Distrib)
{
  static if(is(Distrib == PoissonDistribution!(T)) | is(Distrib == BinomialDistribution!(T)))
    enum bool unitDispsersion = true;
  else
    enum bool unitDispsersion = false;
}

