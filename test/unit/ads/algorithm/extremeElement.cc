// -*- C++ -*-

#include "stlib/ads/algorithm/extremeElement.h"

#include <algorithm>
#include <array>

using namespace stlib;

int
main()
{
  // Even
  {
    typedef std::array<int, 6> Sequence;
    std::array<Sequence, 3> sequences;
    sequences[0] = Sequence{{0, 1, 2, 3, 4, 5}};
    sequences[1] = Sequence{{3, 4, 5, 0, 1, 2}};
    sequences[2] = Sequence{{1, 2, 3, 4, 5, 0}};
    for (std::size_t i = 0; i != sequences.size(); ++i) {
      const Sequence& x = sequences[i];
      assert(ads::findMinimumElementUnrolledEven(x.begin(), x.end()) ==
             std::min_element(x.begin(), x.end()));
      assert(ads::findMinimumElementUnrolled(x.begin(), x.end()) ==
             std::min_element(x.begin(), x.end()));
      assert(ads::findMaximumElementUnrolledEven(x.begin(), x.end()) ==
             std::max_element(x.begin(), x.end()));
      assert(ads::findMaximumElementUnrolled(x.begin(), x.end()) ==
             std::max_element(x.begin(), x.end()));
    }
  }

  // Odd.
  {
    typedef std::array<int, 7> Sequence;
    std::array<Sequence, 3> sequences;
    sequences[0] = Sequence{{0, 1, 2, 3, 4, 5, 6}};
    sequences[1] = Sequence{{3, 4, 5, 6, 0, 1, 2}};
    sequences[2] = Sequence{{1, 2, 3, 4, 5, 6, 0}};
    for (std::size_t i = 0; i != sequences.size(); ++i) {
      const Sequence& x = sequences[i];
      assert(ads::findMinimumElementUnrolledOdd(x.begin(), x.end()) ==
             std::min_element(x.begin(), x.end()));
      assert(ads::findMinimumElementUnrolled(x.begin(), x.end()) ==
             std::min_element(x.begin(), x.end()));
      assert(ads::findMaximumElementUnrolledOdd(x.begin(), x.end()) ==
             std::max_element(x.begin(), x.end()));
      assert(ads::findMaximumElementUnrolled(x.begin(), x.end()) ==
             std::max_element(x.begin(), x.end()));
    }
  }

  return 0;
}
