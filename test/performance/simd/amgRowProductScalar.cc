// -*- C++ -*-

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ext/vector.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>


using namespace stlib;

int
main(int argc, char* argv[])
{
  typedef double Float;

  ads::ParseOptionsArguments parser(argc, argv);

  // The number of rows in the matrix.
  std::size_t nRows = 1024;
  parser.getOption('r', &nRows);

  // The number of off-diagonal elements per row.
  std::size_t nOffDiagonal = 8;
  parser.getOption('o', &nOffDiagonal);
  assert(nOffDiagonal > 0);
  assert(nRows >= nOffDiagonal);

  assert(parser.areOptionsEmpty());
  assert(parser.areArgumentsEmpty());

  // The values for the off-diagonal elements.
  std::vector<Float> v(nRows * nOffDiagonal);
  std::default_random_engine generator;
  std::uniform_real_distribution<Float> uniform(0, 1);
  for (std::size_t i = 0; i != v.size(); ++i) {
    v[i] = uniform(generator);
  }
  // The column indices for the off-diagonal elements.
  std::vector<std::size_t> c(nRows * nOffDiagonal);
  for (std::size_t i = 0; i != c.size(); ++i) {
    c[i] = i % nRows;
  }
  std::random_shuffle(c.begin(), c.end());
  // The off-diagonal index delimiters.
  std::vector<std::size_t> r(nRows + 1);
  for (std::size_t i = 0; i != r.size(); ++i) {
    r[i] = i * nOffDiagonal;
  }
  // The variable.
  std::vector<Float> x(nRows);
  for (std::size_t i = 0; i != x.size(); ++i) {
    x[i] = uniform(generator);
  }

  ads::Timer timer;
  timer.tic();
  for (std::size_t i = 0; i != x.size(); ++i) {
    Float sum = 0;
    for (std::size_t j = r[i]; j != r[i + 1]; ++j) {
      sum += v[j] * x[c[j]];
    }
    x[i] += sum;
  }
  double elapsedTime = timer.toc();

  std::cout << "Meaningless result = " << stlib::ext::sum(x) << std::endl
            << "Scalar time = " << elapsedTime << " seconds." << std::endl
            << "  Time per row = " << elapsedTime / nRows * 1e9
            << " nanoseconds." << std::endl
            << "  Time per term = "
            << elapsedTime / (nRows * nOffDiagonal) * 1e9
            << " nanoseconds." << std::endl;

  return 0;
}
