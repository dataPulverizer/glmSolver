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
import std.stdio : writeln;


/* ldc2 test.d arrays.d arraycommon.d apply.d link.d distributions.d tools.d linearalgebra.d io.d fit.d -O2 -L-lopenblas -L-lpthread -L-llapacke -L-llapack -L-lm && ./test */
void main()
{
  AbstractDistribution!(double) distrib = new GaussianDistribution!(double)();
  AbstractLink!(double) link = new IdentityLink!(double)();
  AbstractSolver!(double) mySolver = new VanillaSolver!(double)();
  auto mu = createRandomColumnVector!(double)(20);
  auto eta = createRandomColumnVector!(double)(20);
  auto y = createRandomColumnVector!(double)(20);
  auto R = createRandomMatrix!(double)(10, 10);
  auto xwx = createRandomMatrix!(double)(10, 10);
  //writeln("Weights: \n", mySolver.W(distrib, link, mu, eta));
  //writeln("Covariance: \n", cov(R, xwx));
  writeln("Weights: \n", mySolver.cov(R, xwx));
}

