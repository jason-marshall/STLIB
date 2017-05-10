// -*- C++ -*-

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ext/vector.h"
#include "stlib/simd/allocator.h"
#include "stlib/simd/functions.h"

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>


using namespace stlib;

int
main(int argc, char* argv[])
{
  typedef double Float;
  //typedef typename simd::Vector<Float>::Type Vector;
  std::size_t const VectorSize = simd::Vector<Float>::Size;

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

  std::size_t const nOffDiagonalPadded =
    (nOffDiagonal + VectorSize - 1) / VectorSize * VectorSize;

  std::default_random_engine generator;
  std::uniform_real_distribution<Float> uniform(0, 1);

  // The values for the off-diagonal elements.
  std::vector<Float, simd::allocator<Float> >
    v(nRows * nOffDiagonalPadded);
  std::fill(v.begin(), v.end(), 0);
  for (std::size_t i = 0; i != nRows; ++i) {
    for (std::size_t j = 0; j != nOffDiagonal; ++j) {
      v[i * nOffDiagonalPadded + j] = uniform(generator);
    }
  }
  // The column indices for the off-diagonal elements.
  std::vector<std::size_t, simd::allocator<std::size_t> >
    c(nRows * nOffDiagonalPadded);
  {
    std::vector<std::size_t> data(nRows * nOffDiagonal);
    for (std::size_t i = 0; i != data.size(); ++i) {
      data[i] = i % nRows;
    }
    std::random_shuffle(data.begin(), data.end());
    // The padded values are used to index into x to obtain the padded value
    // (zero).
    std::fill(c.begin(), c.end(), nRows);
    for (std::size_t i = 0; i != nRows; ++i) {
      for (std::size_t j = 0; j != nOffDiagonal; ++j) {
        c[i * nOffDiagonalPadded + j] = data[i * nOffDiagonal + j];
      }
    }
  }
  // The off-diagonal index delimiters.
  std::vector<std::size_t> r(nRows + 1);
  for (std::size_t i = 0; i != r.size(); ++i) {
    r[i] = i * nOffDiagonalPadded;
  }
  // The variable is padded with a zero at the end.
  std::vector<Float> x(nRows + 1);
  for (std::size_t i = 0; i != nRows; ++i) {
    x[i] = uniform(generator);
  }
  x.back() = 0;

  ads::Timer timer;
  timer.tic();
  ALIGN_SIMD Float xa[VectorSize];
  ALIGN_SIMD Float sa[VectorSize];
  for (std::size_t i = 0; i != nRows; ++i) {
    Float sum = 0;
    for (std::size_t j = r[i]; j != r[i + 1]; j += VectorSize) {
      for (std::size_t k = 0; k != VectorSize; ++k) {
        xa[k] = x[c[j + k]];
      }
      simd::store(sa, simd::load(&v[j]) * simd::load(&xa[0]));
      for (std::size_t k = 0; k != VectorSize; ++k) {
        sum += sa[k];
      }
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
