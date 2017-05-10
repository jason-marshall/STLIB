// -*- C++ -*-

#include "stlib/ads/iterator/SingleValueIterator.h"

#include <cassert>

using namespace stlib;

int
main()
{
  typedef ads::SingleValueIterator<double> Iterator;

  const std::size_t Size = 10;
  const double Value = 7;

  // Forward iterator.
  {
    // Pre-increment.
    Iterator x(Value);
    for (std::size_t i = 0; i != Size; ++i, ++x) {
      assert(*x == Value);
    }

    // Post-increment.
    for (std::size_t i = 0; i != Size; ++i, x++) {
      assert(*x == Value);
    }

    // Post-increment.
    for (std::size_t i = 0; i != Size; ++i) {
      assert(*x++ == Value);
    }
  }

  // Bi-directional iterator.
  {
    // Pre-decrement.
    Iterator x(Value);
    for (std::size_t i = 0; i != Size; ++i, --x) {
      assert(*x == Value);
    }

    // Post-decrement.
    for (std::size_t i = 0; i != Size; ++i, x--) {
      assert(*x == Value);
    }

    // Post-decrement.
    for (std::size_t i = 0; i != Size; ++i) {
      assert(*x-- == Value);
    }

  }
  // Random access iterator.
  {
    // Indexing.
    Iterator x(Value);
    for (std::size_t i = 0; i != Size; ++i) {
      assert(x[i] == Value);
    }

    // Positive offseting.
    for (std::size_t i = 0; i != Size; ++i) {
      Iterator y(Value);
      y += i;
      assert(*y == Value);
    }

    // Positive offseting.
    for (std::size_t i = 0; i != Size; ++i) {
      assert(*(x + i) == Value);
    }

    // Negative offseting.
    for (std::size_t i = 0; i != Size; ++i) {
      Iterator y(Value);
      y -= -i;
      assert(*y == Value);
    }

    // Negative offseting.
    for (std::size_t i = 0; i != Size; ++i) {
      assert(*(x - (-i)) == Value);
    }
  }

  return 0;
}
