/*
  GLM Function(s)
*/
module glmsolverd.fit;

import std.conv: to;
import std.stdio : writeln;
import std.traits: isFloatingPoint, isIntegral, isNumeric;

import std.parallelism;
import std.range : iota;

import std.math : pow;
import std.algorithm: min, max;

import glmsolverd.arrays;
import glmsolverd.common;
import glmsolverd.apply;
import glmsolverd.link;
import glmsolverd.distributions;
import glmsolverd.tools;
import glmsolverd.linearalgebra;

import std.stdio: writeln;

/**************************************** GLM Function ***************************************/
auto glm(T, CBLAS_LAYOUT layout = CblasColMajor)(
        RegularData dataType,  Matrix!(T, layout) x, 
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
  ColumnVector!(T) w;
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
/*
  For now Block1D overload is limited only to the following solvers ...

  GESVSolver
  POSVSolver
  SYSVSolver
*/
auto glm(T, CBLAS_LAYOUT layout = CblasColMajor)(
        Block1D dataType, Matrix!(T, layout)[] x, 
        Matrix!(T, layout)[] _y, AbstractDistribution!T distrib, AbstractLink!T link,
        AbstractSolver!(T) solver = new VanillaSolver!(T)(), 
        AbstractInverse!(T, layout) inverse = new GETRIInverse!(T, layout)(), 
        Control!T control = new Control!T(), ColumnVector!(T)[] offset = new ColumnVector!(T)[0],
        ColumnVector!(T)[] weights = new ColumnVector!(T)[0])
if(isFloatingPoint!T)
{
  auto nBlocks = _y.length;
  auto init = distrib.init(_y, weights);
  ColumnVector!(T)[] y = init[0]; 
  ColumnVector!(T)[] mu = init[1]; weights = init[2];
  auto eta = link.linkfun(mu);

  auto coef = zerosColumn!T(x[0].ncol);
  auto coefold = zerosColumn!T(x[0].ncol);

  auto absErr = T.infinity;
  auto relErr = T.infinity;
  ColumnVector!(T)[] residuals;
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
  ColumnVector!(T)[] w;
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

/**************************************** BLOCK PARALLEL GLM Function ***************************************/
/*
  For now Block1DParallel overload is limited only to the following solvers ...

  GESVSolver
  POSVSolver
  SYSVSolver
*/

auto glm(T, CBLAS_LAYOUT layout = CblasColMajor)(
        Block1DParallel dataType, Matrix!(T, layout)[] x, 
        Matrix!(T, layout)[] _y, AbstractDistribution!T distrib, AbstractLink!T link,
        AbstractSolver!(T) solver = new VanillaSolver!(T)(), 
        AbstractInverse!(T, layout) inverse = new GETRIInverse!(T, layout)(), 
        Control!T control = new Control!T(), ColumnVector!(T)[] offset = new ColumnVector!(T)[0],
        ColumnVector!(T)[] weights = new ColumnVector!(T)[0])
if(isFloatingPoint!T)
{
  openblas_set_num_threads(1);
  auto nBlocks = _y.length;
  auto init = distrib.init(dataType, _y, weights);
  ColumnVector!(T)[] y = init[0]; 
  ColumnVector!(T)[] mu = init[1]; weights = init[2];
  auto eta = link.linkfun(dataType, mu);

  auto coef = zerosColumn!T(x[0].ncol);
  auto coefold = zerosColumn!T(x[0].ncol);

  auto absErr = T.infinity;
  auto relErr = T.infinity;
  ColumnVector!(T)[] residuals; // = zerosColumn!T(y.len);
  auto dev = T.infinity;
  auto devold = T.infinity;

  ulong iter = 1;
  ulong n = 0;

  auto nStore = taskPool.workerLocalStorage(0L);
  /* Parallelised reduction required */
  foreach(i; taskPool.parallel(iota(nBlocks)))
    nStore.get += x[i].nrow;
  foreach (_n; nStore.toRange)
        n += _n;

  auto p = x[0].ncol;
  bool converged, badBreak, doOffset, doWeights;

  if(offset.length != 0)
    doOffset = true;
  if(weights.length != 0)
    doWeights = true;

  Matrix!(T, layout) cov, xw, xwx, R;
  ColumnVector!(T)[] w;
  while(relErr > control.epsilon)
  {
    if(control.printError)
      writeln("Iteration: ", iter);
    auto z = Z!(double)(dataType, link, y, mu, eta);
    if(doOffset)
    {
      foreach(i; taskPool.parallel(iota(nBlocks)))
        z[i] = map!( (x1, x2) => x1 - x2 )(z[i], offset[i]);
    }
    
    /* Weights calculation standard vs sqrt */
    w = solver.W(dataType, distrib, link, mu, eta);
    
    if(doWeights)
    {
      foreach(i; taskPool.parallel(iota(nBlocks)))
        w[i] = map!( (x1, x2) => x1*x2 )(w[i], weights[i]);
    }

    solver.solve(dataType, R, xwx, xw, x, z, w, coef);
    
    if(control.printCoef)
      writeln(coef);
    
    foreach(i; taskPool.parallel(iota(nBlocks)))
        eta[i] = mult_(x[i], coef);
    
    if(doOffset)
    {
      foreach(i; taskPool.parallel(iota(nBlocks)))
        eta[i] += offset[i];
    }
    
    mu = link.linkinv(dataType, eta);

    if(weights.length == 0)
      residuals = distrib.devianceResiduals(dataType, mu, y);
    else
      residuals = distrib.devianceResiduals(dataType, mu, y, weights);
    
    auto devStore = taskPool.workerLocalStorage(cast(T)0);
    dev = cast(T)0;
    foreach(i; taskPool.parallel(iota(nBlocks)))
      devStore.get += sum!T(residuals[i]);
    foreach (_dev; devStore.toRange)
      dev += _dev;

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
      
      /*
        Do something about multiple thread reference to coef
      */
      foreach(i; taskPool.parallel(iota(nBlocks)))
        eta[i] = mult_(x[i], coef);
      
      if(doOffset)
      {
        foreach(i; taskPool.parallel(iota(nBlocks)))
          eta[i] += offset[i];
      }
      mu = link.linkinv(eta);

      if(weights.length == 0)
        residuals = distrib.devianceResiduals(mu, y);
      else
        residuals = distrib.devianceResiduals(mu, y, weights);
      
      devStore = taskPool.workerLocalStorage(cast(T)0);
      dev = cast(T)0;
      /* Parallel reduction required */
      foreach(i; taskPool.parallel(iota(nBlocks)))
        devStore.get += sum!T(residuals[i]);
      foreach (_dev; devStore.toRange)
        dev += _dev;

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

  openblas_set_num_threads(cast(int)totalCPUs);
  auto obj = new GLM!(T, layout)(iter, converged, phi, distrib, link, coef, cov, dev, absErr, relErr);
  return obj;
}

/**************************************** Gradient Descent GLM ***************************************/

auto glm(T, CBLAS_LAYOUT layout = CblasColMajor)(
        RegularData dataType,  Matrix!(T, layout) x, 
        Matrix!(T, layout) _y, AbstractDistribution!T distrib, AbstractLink!T link,
        AbstractGradientSolver!(T) solver, 
        AbstractInverse!(T, layout) inverse = new GETRIInverse!(T, layout)(), 
        Control!T control = new Control!T(), ColumnVector!T offset = zerosColumn!T(0),
        ColumnVector!T weights = zerosColumn!T(0))
if(isFloatingPoint!T)
{
  auto init = distrib.init(_y, weights);
  auto y = init[0]; auto mu = init[1]; weights = init[2];
  auto eta = link.linkfun(mu);

  //auto coef = zerosColumn!T(x.ncol);
  auto n = x.nrow; auto p = x.ncol;
  auto coef = sampleStandardNormal!T(x.ncol)/p;//pow(p, 0.5);
  auto coefold = zerosColumn!T(x.ncol);

  auto absErr = T.infinity;
  auto relErr = T.infinity;
  auto residuals = zerosColumn!T(y.len);
  auto dev = T.infinity;
  auto devold = T.infinity;

  ulong iter = 1;
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

    solver.solve(distrib, link, y, x, mu, eta, coef);
    
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
      //writeln("Entered step control");
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

    if(iter % 100 == 0)
      writeln("Deviance: ", dev, ", iteration: ", iter);

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
  
  auto z = link.Z(y, mu, eta);
  
  if(doOffset)
    z = map!( (x1, x2) => x1 - x2 )(z, offset);
  
  /* Weights calculation standard vs sqrt */
  w = solver.W(distrib, link, mu, eta);
  
  if(doWeights)
    w = map!( (x1, x2) => x1*x2 )(w, weights);
  
  writeln("Coefficients: ", coef.getData);

  solver.XWX(xwx, xw, x, z, w);
  
  cov = solver.cov(inverse, R, xwx, xw);
  T phi = 1;

  if(!unitDispsersion!(T, typeof(distrib)))
  {
    phi = dev/(n - p);
    imap!( (T x) => x*phi)(cov);
  }

  auto obj = new GLM!(T, layout)(iter, converged, phi, distrib, link, coef, cov, dev, absErr, relErr);
  return obj;
}
