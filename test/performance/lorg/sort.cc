// -*- C++ -*-

#include "stlib/lorg/sort.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/ext/vector.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>

using namespace stlib;

template<typename _Integer>
void
time(const int digits = std::numeric_limits<_Integer>::digits)
{
  _Integer mask;
  if (digits == std::numeric_limits<_Integer>::digits) {
    mask = -1;
  }
  else {
    mask = (_Integer(1) << digits) - 1;
  }

  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;

  ads::Timer timer;
  double elapsedTime;
  std::size_t result = 0;

  std::cout << "size,introsort,RCI\n";
  for (std::size_t exponent = 10; exponent <= 22; ++exponent) {
    std::vector<_Integer> codes(1 << exponent);
    std::cout << codes.size();
    for (std::size_t i = 0; i != codes.size(); ++i) {
      codes[i] = generator() & mask;
    }
    {
      std::vector<std::pair<_Integer, std::size_t> > pairs(codes.size());
      for (std::size_t i = 0; i != pairs.size(); ++i) {
        pairs[i].first = codes[i];
        pairs[i].second = i;
      }
      timer.tic();
      std::sort(pairs.begin(), pairs.end());
      elapsedTime = timer.toc();
      for (std::size_t i = 0; i != pairs.size(); ++i) {
        result += pairs[i].first + pairs[i].second;
      }
      // Nanoseconds per element.
      std::cout << ',' << 1e9 * elapsedTime / pairs.size();
    }
    {
      std::vector<std::pair<_Integer, std::size_t> > pairs(codes.size());
      for (std::size_t i = 0; i != pairs.size(); ++i) {
        pairs[i].first = codes[i];
        pairs[i].second = i;
      }
      timer.tic();
      lorg::RciSort<_Integer, std::size_t> rci(&pairs, digits);
      rci.sort();
      elapsedTime = timer.toc();
      for (std::size_t i = 0; i != pairs.size(); ++i) {
        result += pairs[i].first + pairs[i].second;
      }
      // Nanoseconds per element.
      std::cout << ',' << 1e9 * elapsedTime / pairs.size();
    }
    std::cout << '\n';
  }

  std::cout << "Meaningless result = " << result << "\n";
}

int
main()
{
  // Specify the number of digits.
  for (int digits = std::numeric_limits<std::size_t>::digits; digits >= 8;
       digits /= 2) {
    std::cout << digits << " of " << std::numeric_limits<std::size_t>::digits
              << " binary digits:\n";
    time<std::size_t>(digits);
  }

  // Use all of the digits.
  std::cout << std::numeric_limits<std::size_t>::digits << " binary digits:\n";
  time<std::size_t>();
  std::cout << std::numeric_limits<unsigned>::digits << " binary digits:\n";
  time<unsigned>();
  std::cout << std::numeric_limits<unsigned short>::digits
            << " binary digits:\n";
  time<unsigned short>();
  std::cout << std::numeric_limits<unsigned char>::digits
            << " binary digits:\n";
  time<unsigned char>();

  return 0;
}
