/*
  This file is for the glm link functions
*/
module glmsolverd.link;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;

import std.conv: to;
import std.algorithm: min, max, fold;
import std.math: atan, exp, expm1, log, modf, fabs, fmax, fmin, cos, tan, PI;
import std.mathspecial : normalDistribution, normalDistributionInverse;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

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
}
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
}

