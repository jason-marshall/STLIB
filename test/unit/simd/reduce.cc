// -*- C++ -*-

#include "stlib/simd/reduce.h"
#include "stlib/simd/allocator.h"

#include <vector>


template<typename _Float>
inline
void
test()
{
  std::size_t const VectorSize = stlib::simd::Vector<_Float>::Size;
  {
    std::vector<_Float, stlib::simd::allocator<_Float> > data(10 * VectorSize);
    // 0 1 2 3 ...
    for (std::size_t i = 0; i != data.size(); ++i) {
      data[i] = i;
    }
    assert(stlib::simd::minAlignedPadded(&*data.begin(), &*data.end()) == 0);
    // ... 3 2 1 0
    for (std::size_t i = 0; i != data.size(); ++i) {
      data[i] = data.size() - i - 1;
    }
    assert(stlib::simd::minAlignedPadded(&*data.begin(), &*data.end()) == 0);
  }
}


int
main()
{
  test<float>();
  test<double>();

  return 0;
}
