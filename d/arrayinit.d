/*
  This benchmark shows that array appending is slow (as expected!), but D's `new T[n]`
  initialization is not "slow", but you can save up to ~ 25% of the time by using malloc.
  Anyway the code is now here if I ever need to use it. It's something to consider for any
  array-heavy numeric computation. D's arrays might not be super-efficient. Try with both
  ldc and dmd compilers:

  # Bounds check off makes no difference
  dmd arrayinit.d -O -boundscheck=off  && ./arrayinit
  ldc2 arrayinit.d -O2 --boundscheck=off  && ./arrayinit

  One of the other takeaways is that the ldc compiler produces MUCH faster executables.
  Once you start doing work inside the loop, the initialization time pales in 
  significance so it definitely depends on what you are doing while you are initializing
  the array. If you are initializing massive array with a single number, it might be
  worth using the "unsafe" method to eek that bit more performance out.
*/
import std.stdio : writeln;

/* Default way D initializes arrays */
T[] defaultInit(T)(T x, ulong n)
{
  auto arr = new T[n];
  for(ulong i = 0; i < n; ++i)
    arr[i] = x;
  return arr;
}

/* Initialization by appending array */
T[] appendInit(T)(T x, ulong n)
{
  T[] arr;
  for(ulong i = 0; i < n; ++i)
    arr ~= x;
  return arr;
}

import core.stdc.stdlib: malloc;
/* Initialize array using a pointer */
T[] unsafeInit(T)(T x, ulong n)
{
  auto arr = cast(T*)malloc(T.sizeof*n);
  if(arr == null)
    assert(0, "Array Allocation Failed!");
  for(ulong i = 0; i < n; ++i)
    arr[i] = x;
  return arr[0..n];
}
/* What if I return a raw pointer instead of an array? */
auto unsafeInit2(T)(T x, ulong n)
{
  auto arr = cast(T*)malloc(T.sizeof*n);
  if(arr == null)
    assert(0, "Array Allocation Failed!");
  for(ulong i = 0; i < n; ++i)
    arr[i] = x;
  return arr;
}

/* Create a random array using unsafe method */
import std.random : Mt19937_64, unpredictableSeed, uniform01;
T[] createRandomArray(T)(ulong n)
{
  Mt19937_64 gen;
  gen.seed(unpredictableSeed);

  auto arr = new T[n];
  for(ulong i = 0; i < n; ++i)
    arr[i] = uniform01!(T)(gen);
  return arr;
}
T[] createUnsafeRandomArray(T)(ulong n)
{
  Mt19937_64 gen;
  gen.seed(unpredictableSeed);

  auto arr = cast(T*)malloc(T.sizeof * n);
  if(arr == null)
    assert(0, "Array Allocation Failed!");
  
  for(ulong i = 0; i < n; ++i)
    arr[i] = uniform01!(T)(gen);
  return arr[0..n];
}
/*
  Apply a function to all the natural 1..(n + 1) numbers 
  and return the array.
*/
T[] applyFunction(alias fun, T)(ulong n)
{
  auto arr = new T[n];
  for(ulong i = 0; i < n; ++i)
    arr[i] = cast(T)fun(i + 1);
  return arr;
}
T[] unsafeApplyFunction(alias fun, T)(ulong n)
{
  auto arr = cast(T*)malloc(T.sizeof * n);
  if(arr == null)
    assert(0, "Array Allocation Failed");
  for(ulong i = 0; i < n; ++i)
    arr[i] = cast(T)fun(i + 1);
  return arr[0..n];
}

/* 
  Benchmarks ...
*/
import std.datetime.stopwatch : AutoStart, StopWatch;
import std.math: log;

void main()
{
  /* Make sure that you have enough system memory to run this code */
  ulong n = 10_000_000;

  /* Append Init Timings */
  auto sw = StopWatch(AutoStart.no);
  sw.start();
  auto y1 = appendInit!(double)(1, n);
  sw.stop();
  writeln("Append init time (ms): ", sw.peek.total!"msecs");
  sw.reset();
  writeln("Print 5 elements: ", y1[(n-5)..n]);

  /* Default Init Timings */
  sw = StopWatch(AutoStart.no);
  sw.start();
  auto x = defaultInit!(double)(1, n);
  sw.stop();
  writeln("Default init time (ms): ", sw.peek.total!"msecs");
  sw.reset();
  /* Printing forces compiler optimization to form the vector */
  writeln("Print 5 elements: ", x[(n-5)..n]);

  /* Create array using pointer */
  sw = StopWatch(AutoStart.no);
  sw.start();
  auto y2 = unsafeInit!(double)(1, n);
  sw.stop();
  writeln("Pointer init time (ms): ", sw.peek.total!"msecs");
  sw.reset();
  writeln("Print 5 elements: ", y2[(n-5)..n]);

  /* Does returning a pointer rather than an array make a difference? (No) */
  sw = StopWatch(AutoStart.no);
  sw.start();
  auto y3 = unsafeInit2!(double)(1, n);
  sw.stop();
  writeln("Returning pointer makes no difference: ", sw.peek.total!"msecs");
  sw.reset();
  writeln("Print 5 elements: ", y3[(n-5)..n]);

  /* Create random array conventionally */
  sw = StopWatch(AutoStart.no);
  sw.start();
  auto x1 = createRandomArray!(double)(n);
  sw.stop();
  writeln("Time (ms) for conventional random array: ", sw.peek.total!"msecs");
  sw.reset();
  writeln("Print 5 elements: ", x1[(n-5)..n]);

  /* Create random array using unsafe method */
  sw = StopWatch(AutoStart.no);
  sw.start();
  auto x2 = createUnsafeRandomArray!(double)(n);
  sw.stop();
  writeln("Time (ms) for random array in an unsafe way: ", sw.peek.total!"msecs");
  sw.reset();
  writeln("Print 5 elements: ", x2[(n-5)..n]);

  /* Apply function benchmarks */
  sw = StopWatch(AutoStart.no);
  sw.start();
  auto a1 = applyFunction!(log, double)(n);
  sw.stop();
  writeln("Time (ms) for function apply: ", sw.peek.total!"msecs");
  sw.reset();
  writeln("Print 5 elements: ", a1[0..5]);

  /* Unsafe apply function benchmarks */
  sw = StopWatch(AutoStart.no);
  sw.start();
  auto a2 = unsafeApplyFunction!(log, double)(n);
  sw.stop();
  writeln("Time (ms) unsafe function apply: ", sw.peek.total!"msecs");
  sw.reset();
  writeln("Print 5 elements: ", a2[0..5]);
}

