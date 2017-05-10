// -*- C++ -*-

#include "stlib/ads/timer/Timer.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace stlib;

int
main()
{
  typedef std::pair<std::size_t, double> Pair;
  typedef std::vector<Pair> SparseVector;
  typedef SparseVector::const_iterator const_iterator;

  ads::Timer timer;
  std::size_t size = 100;
  std::vector<double> x(size, 0);
  SparseVector s(size);
  for (std::size_t i = 0; i != s.size(); ++i) {
    s[i] = Pair(i, double(i));
  }
  std::random_shuffle(s.begin(), s.end());
  const std::size_t Count = 1000000;

  // Warm up.
  for (std::size_t n = 0; n != 1000; ++n) {
    for (const_iterator i = s.begin(); i != s.end(); ++i) {
      x[i->first] += i->second;
    }
  }

  timer.tic();
  for (std::size_t n = 0; n != Count; ++n) {
    for (const_iterator i = s.begin(); i != s.end(); ++i) {
      x[i->first] += i->second;
    }
  }
  double elapsedTime = timer.toc();

  std::cout
      << "Elapsed time = " << elapsedTime << '\n'
      << "CLOCKS_PER_SEC = " << CLOCKS_PER_SEC << "\n"
      << "Time per operation = " << elapsedTime / (s.size() * Count) * 1e9
      << " nanoseconds.\n"
      << "Meaningless result = " << std::accumulate(x.begin(), x.end(), 0.)
      << "\n";

  return 0;
}
