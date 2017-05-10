// -*- C++ -*-

#include <type_traits>
#include <vector>
#include <iostream>
#include <xmmintrin.h>
#include <cassert>

int
main()
{
  typedef std::aligned_storage<sizeof(__m128),
          std::alignment_of<__m128>::value>::type m128;

  assert(sizeof(__m128) == 4 * sizeof(float));
  assert(std::alignment_of<__m128>::value == sizeof(__m128));
  for (std::size_t i = 1; i != 10; ++i) {
    std::vector<m128> x(i);
    // Note that the following will not always work. The allocator may not
    // respect the 16-byte alignment.
    //assert(std::size_t(&x[0]) % 16 == 0);
  }
  return 0;
}
