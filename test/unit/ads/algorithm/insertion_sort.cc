// -*- C++ -*-

#include "stlib/ads/algorithm/insertion_sort.h"

#include <algorithm>
#include <functional>
#include <vector>

#include <cassert>

using namespace stlib;

int main()
{
  std::vector<double> v(1000);
  for (std::size_t i = 0; i != v.size(); ++i) {
    v[i] = i;
  }
  std::random_shuffle(v.begin(), v.end());

  ads::insertion_sort(v.begin(), v.end());
  assert(std::is_sorted(v.begin(), v.end()));

  ads::insertion_sort(v.begin(), v.end(), std::greater<double>());
  assert(std::is_sorted(v.begin(), v.end(), std::greater<double>()));
}
