/*
  GLM Function(s)
*/
module fit;

import std.conv: to;
import std.stdio : writeln;
import std.traits: isFloatingPoint, isIntegral, isNumeric;
import arrays;
import arraycommon;
import apply;
import link;
import distributions;
import tools;
import linearalgebra;

/**************************************** GLM Function ***************************************/
auto glm(T, CBLAS_LAYOUT layout = CblasColMajor)(Matrix!(T, layout) x, 
        Matrix!(T, layout) _y, AbstractDistribution!T distrib, AbstractLink!T link,
        AbstractSolver!(T) qrSolver = new QRSolver!(T)(), Control!T control = new Control!T(),
        ColumnVector!T offset = zerosColumn!T(0), ColumnVector!T weights = zerosColumn!T(0))
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

  Matrix!(T, layout) cov, xwx, R;
  ColumnVector!(T) w, wy;
  while(relErr > control.epsilon)
  {
    if(control.printError)
      writeln("Iteration: ", iter);
    auto z = link.Z(y, mu, eta);
    if(doOffset)
      z = map!( (x1, x2) => x1 - x2 )(z, offset);
    
    /* Weights calculation standard vs sqrt */
    //if(qrSolver)
    //{
    //  w = WS(distrib, link, mu, eta);
    //}else{
    //  w = W(distrib, link, mu, eta);
    //}
    w = qrSolver.W(distrib, link, mu, eta);
    
    if(doWeights)
      w = map!( (x1, x2) => x1*x2 )(w, weights);

    //if(qrSolver)
    //{
    //  _qr_solver2!(T, layout)(R, x, z, w, coef);
    //}else{
    //  _conventional_solver!(T, layout)(xwx, x, z, w, coef);
    //}
    qrSolver.solve(R, xwx, x, z, w, coef);
    
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
  //if(qrSolver)
  //{
  //  xwx = mult_!(T, layout, CblasTrans)(R, R.dup);
  //  cov = inv(xwx);
  //}else{
  //  cov = inv(xwx);
  //}
  cov = qrSolver.cov(R, xwx);
  auto obj = new GLM!(T, layout)(iter, converged, distrib, link, coef, cov, dev, absErr, relErr);
  return obj;
}

/********************************************************************************/
auto glm_old(T, CBLAS_LAYOUT layout = CblasColMajor)(Matrix!(T, layout) x, 
        Matrix!(T, layout) _y, AbstractDistribution!T distrib, AbstractLink!T link,
        bool qrSolver = true, Control!T control = new Control!T(),
        ColumnVector!T offset = zerosColumn!T(0), ColumnVector!T weights = zerosColumn!T(0))
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

  Matrix!(T, layout) cov, xwx, R;
  ColumnVector!(T) w, wy;
  while(relErr > control.epsilon)
  {
    if(control.printError)
      writeln("Iteration: ", iter);
    auto z = link.Z(y, mu, eta);
    if(doOffset)
      z = map!( (x1, x2) => x1 - x2 )(z, offset);
    
    /* Weights calculation standard vs sqrt */
    if(qrSolver)
    {
      w = WS(distrib, link, mu, eta);
    }else{
      w = W(distrib, link, mu, eta);
    }
    
    if(doWeights)
      w = map!( (x1, x2) => x1*x2 )(w, weights);

    if(qrSolver)
    {
      _qr_solver2!(T, layout)(R, x, z, w, coef);
    }else{
      _conventional_solver!(T, layout)(xwx, x, z, w, coef);
    }
    
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
  if(qrSolver)
  {
    xwx = mult_!(T, layout, CblasTrans)(R, R.dup);
    cov = inv(xwx);
  }else{
    cov = inv(xwx);
  }
  auto obj = new GLM!(T, layout)(iter, converged, distrib, link, coef, cov, dev, absErr, relErr);
  return obj;
}

