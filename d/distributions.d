/*
  Module for distributions for GLM
*/
module distributions;
import apply;
import arrays;
import arraycommon;
import std.conv: to;
import std.math: log;
import std.typecons: Tuple, tuple;

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
}
T y_log_y(T)(T y, T x)
{
  //pragma(inline, true);
  return y != 0 ? y * log(y/x) : 0;
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
}

/* Evaluates if the dispersion should be 1 */
template unitDispsersion(T, Distrib)
{
  static if(is(Distrib == PoissonDistribution!(T)) | is(Distrib == BinomialDistribution!(T)))
    enum bool unitDispsersion = true;
  else
    enum bool unitDispsersion = false;
}

