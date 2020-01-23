/*
  GLM Function(s)
*/
module glmsolverd.fit;

import std.conv: to;
import std.stdio : writeln;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;
import glmsolverd.link;
import glmsolverd.distributions;
import glmsolverd.tools;
import glmsolverd.linearalgebra;

/**************************************** GLM Function ***************************************/
auto glm(T, CBLAS_LAYOUT layout = CblasColMajor)(Matrix!(T, layout) x, 
        Matrix!(T, layout) _y, AbstractDistribution!T distrib, AbstractLink!T link,
        AbstractSolver!(T) solver = new VanillaSolver!(T)(), 
        AbstractInverse!(T, layout) inverse = new GETRIInverse!(T, layout)(), 
        Control!T control = new Control!T(), ColumnVector!T offset = zerosColumn!T(0),
        ColumnVector!T weights = zerosColumn!T(0))
if(isFloatingPoint!T)
{
  auto init = distrib.init(_y, weights);
  auto y = init[0]; auto mu = init[1]; weights = init[2];
  auto eta = link.linkfun(mu);

  auto coef = zerosColumn!T(x.ncol);
  auto coefold = zerosColumn!T(x.ncol);

  auto absErr = T.infinity;
  auto relErr = T.infinity;
  auto residuals = zerosColumn!T(y.len);
  auto dev = T.infinity;
  auto devold = T.infinity;

  ulong iter = 1;
  auto n = x.nrow; auto p = x.ncol;
  bool converged, badBreak, doOffset, doWeights;

  if(offset.len != 0)
    doOffset = true;
  if(weights.len != 0)
    doWeights = true;

  Matrix!(T, layout) cov, xw, xwx, R;
  ColumnVector!(T) w;//, wy;
  while(relErr > control.epsilon)
  {
    if(control.printError)
      writeln("Iteration: ", iter);
    auto z = link.Z(y, mu, eta);
    if(doOffset)
      z = map!( (x1, x2) => x1 - x2 )(z, offset);
    
    /* Weights calculation standard vs sqrt */
    w = solver.W(distrib, link, mu, eta);
    
    if(doWeights)
      w = map!( (x1, x2) => x1*x2 )(w, weights);

    solver.solve(R, xwx, xw, x, z, w, coef);
    
    if(control.printCoef)
      writeln(coef);
    
    eta = mult_(x, coef);
    if(doOffset)
      eta += offset;
    mu = link.linkinv(eta);

    if(weights.len == 0)
      residuals = distrib.devianceResiduals(mu, y);
    else
      residuals = distrib.devianceResiduals(mu, y, weights);
    
    dev = sum!T(residuals);

    absErr = absoluteError(dev, devold);
    relErr = relativeError(dev, devold);

    T frac = 1;
    auto coefdiff = map!( (x1, x2) => x1 - x2 )(coef, coefold);

    //Step Control
    while(dev > (devold + control.epsilon*dev))
    {
      if(control.printError)
      {
        writeln("\tStep control");
        writeln("\tFraction: ", frac);
        writeln("\tDeviance: ", dev);
        writeln("\tAbsolute Error: ", absErr);
        writeln("\tRelative Error: ", relErr);
      }
      frac *= 0.5;
      coef = map!( (T x1, T x2) => x1 + (x2 * frac) )(coefold, coefdiff);

      if(control.printCoef)
        writeln(coef);
      
      eta = mult_(x, coef);
      if(doOffset)
        eta += offset;
      mu = link.linkinv(eta);

      if(weights.len == 0)
        residuals = distrib.devianceResiduals(mu, y);
      else
        residuals = distrib.devianceResiduals(mu, y, weights);
      
      dev = sum!T(residuals);

      absErr = absoluteError(dev, devold);
      relErr = relativeError(dev, devold);

      if(frac < control.minstep)
        assert(0, "Step control exceeded.");
    }
    devold = dev;
    coefold = coef.dup;

    if(control.printError)
    {
      writeln("\tDeviance: ", dev);
      writeln("\tAbsolute Error: ", absErr);
      writeln("\tRelative Error: ", relErr);
    }
    if(iter >= control.maxit)
    {
      writeln("Maximum number of iterations " ~ to!string(control.maxit) ~ " has been reached.");
      badBreak = true;
      break;
    }
    iter += 1;
  }
  if(badBreak)
    converged = false;
  else
    converged = true;
  
  cov = solver.cov(inverse, R, xwx, xw);
  T phi = 1;
  if(!unitDispsersion!(T, typeof(distrib)))
  {
    phi = dev/(n - p);
    //inplace alteration of covariance matrix
    imap!( (T x) => x*phi)(cov);
  }

  auto obj = new GLM!(T, layout)(iter, converged, phi, distrib, link, coef, cov, dev, absErr, relErr);
  return obj;
}
/**************************************** BLOCK GLM Function ***************************************/
auto blockGLM(T, CBLAS_LAYOUT layout = CblasColMajor)(BlockMatrix!(T, layout) x, 
        BlockMatrix!(T, layout) _y, AbstractDistribution!T distrib, AbstractLink!T link,
        AbstractSolver!(T) solver = new VanillaSolver!(T)(), 
        AbstractInverse!(T, layout) inverse = new GETRIInverse!(T, layout)(), 
        Control!T control = new Control!T(), BlockColumnVector!T offset = new ColumnVector!(T)[0],
        BlockColumnVector!T weights = new ColumnVector!(T)[0])
if(isFloatingPoint!T)
{
  auto nBlocks = _y.length;
  auto init = distrib.init(_y, weights);
  BlockColumnVector!(T) y = init[0]; 
  BlockColumnVector!(T) mu = init[1]; weights = init[2];
  auto eta = link.linkfun(mu);

  auto coef = zerosColumn!T(x[0].ncol);
  auto coefold = zerosColumn!T(x[0].ncol);

  auto absErr = T.infinity;
  auto relErr = T.infinity;
  BlockColumnVector!(T) residuals; // = zerosColumn!T(y.len);
  auto dev = T.infinity;
  auto devold = T.infinity;

  ulong iter = 1;
  ulong n = 0;
  for(ulong i = 0; i < nBlocks; ++i)
    n += x[i].nrow;
  auto p = x[0].ncol;
  bool converged, badBreak, doOffset, doWeights;

  if(offset.length != 0)
    doOffset = true;
  if(weights.length != 0)
    doWeights = true;

  Matrix!(T, layout) cov, xw, xwx, R;
  BlockColumnVector!(T) w;
  while(relErr > control.epsilon)
  {
    if(control.printError)
      writeln("Iteration: ", iter);
    auto z = Z!(double)(link, y, mu, eta);
    if(doOffset)
    {
      for(ulong i = 0; i < nBlocks; ++i)
        z[i] = map!( (x1, x2) => x1 - x2 )(z[i], offset[i]);
    }
    
    /* Weights calculation standard vs sqrt */
    w = solver.W(distrib, link, mu, eta);
    
    if(doWeights)
    {
      for(ulong i = 0; i < nBlocks; ++i)
        w[i] = map!( (x1, x2) => x1*x2 )(w[i], weights[i]);
    }

    solver.solve(R, xwx, xw, x, z, w, coef);
    
    if(control.printCoef)
      writeln(coef);
    
    for(ulong i = 0; i < nBlocks; ++i)
      eta[i] = mult_(x[i], coef);
    
    if(doOffset)
    {
      for(ulong i = 0; i < nBlocks; ++i)
        eta[i] += offset[i];
    }
    
    mu = link.linkinv(eta);

    if(weights.length == 0)
      residuals = distrib.devianceResiduals(mu, y);
    else
      residuals = distrib.devianceResiduals(mu, y, weights);
    
    //dev = sum!T(residuals);
    dev = cast(T)0;
    for(ulong i = 0; i < nBlocks; ++i)
      dev += sum!T(residuals[i]);

    absErr = absoluteError(dev, devold);
    relErr = relativeError(dev, devold);

    T frac = 1;
    auto coefdiff = map!( (x1, x2) => x1 - x2 )(coef, coefold);

    //Step Control
    while(dev > (devold + control.epsilon*dev))
    {
      if(control.printError)
      {
        writeln("\tStep control");
        writeln("\tFraction: ", frac);
        writeln("\tDeviance: ", dev);
        writeln("\tAbsolute Error: ", absErr);
        writeln("\tRelative Error: ", relErr);
      }
      frac *= 0.5;
      coef = map!( (T x1, T x2) => x1 + (x2 * frac) )(coefold, coefdiff);

      if(control.printCoef)
        writeln(coef);
      
      for(ulong i = 0; i < nBlocks; ++i)
        eta[i] = mult_(x[i], coef);
      if(doOffset)
      {
        for(ulong i = 0; i < nBlocks; ++i)
          eta[i] += offset[i];
      }
      mu = link.linkinv(eta);

      if(weights.length == 0)
        residuals = distrib.devianceResiduals(mu, y);
      else
        residuals = distrib.devianceResiduals(mu, y, weights);
      
      dev = cast(T)0;
      for(ulong i = 0; i < nBlocks; ++i)
        dev = sum!T(residuals[i]);

      absErr = absoluteError(dev, devold);
      relErr = relativeError(dev, devold);

      if(frac < control.minstep)
        assert(0, "Step control exceeded.");
    }
    devold = dev;
    coefold = coef.dup;

    if(control.printError)
    {
      writeln("\tDeviance: ", dev);
      writeln("\tAbsolute Error: ", absErr);
      writeln("\tRelative Error: ", relErr);
    }
    if(iter >= control.maxit)
    {
      writeln("Maximum number of iterations " ~ to!string(control.maxit) ~ " has been reached.");
      badBreak = true;
      break;
    }
    iter += 1;
  }
  if(badBreak)
    converged = false;
  else
    converged = true;
  
  cov = solver.cov(inverse, R, xwx, xw);
  T phi = 1;
  if(!unitDispsersion!(T, typeof(distrib)))
  {
    phi = dev/(n - p);
    //inplace alteration of covariance matrix
    imap!( (T x) => x*phi)(cov);
  }

  auto obj = new GLM!(T, layout)(iter, converged, phi, distrib, link, coef, cov, dev, absErr, relErr);
  return obj;
}
