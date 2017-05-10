// -*- C++ -*-

#include "stlib/simd/align.h"

#include <cassert>
#include <cstddef>

// This class may only be used for stack allocation. If it is allocated on the
// heap, center may not have the correct allignment.
struct ALIGNAS(32) Aligned {
};


int
main()
{
  {
    ALIGNAS(32) char x;
    assert(std::size_t(&x) % 32 == 0);
  }
  {
    ALIGNAS(32) char x[1];
    assert(std::size_t(x) % 32 == 0);
  }
  {
    Aligned x;
    assert(std::size_t(&x) % 32 == 0);
  }
  {
    ALIGN_SIMD char x;
    assert(stlib::simd::isAligned(&x));
  }

  return 0;
}
