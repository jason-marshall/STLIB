// -*- C++ -*-

#include "stlib/ads/iterator/IntIterator.h"

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace ads;

  const std::size_t size = 10;

  // Forward iterator.
  {
    // Pre-increment.
    IntIterator<> x(0);
    for (std::size_t i = 0; i != size; ++i, ++x) {
      assert(i == *x);
    }

    // Post-increment.
    x = 0;
    for (std::size_t i = 0; i != size; ++i, x++) {
      assert(i == *x);
    }

    // Post-increment.
    x = 0;
    for (std::size_t i = 0; i != size; ++i) {
      assert(i == *x++);
    }
  }

  // Bi-directional iterator.
  {
    // Pre-decrement.
    IntIterator<> x(size - 1);
    for (int i = size - 1; i >= 0; --i, --x) {
      assert(i == int(*x));
    }

    // Post-decrement.
    x = size - 1;
    for (int i = size - 1; i >= 0; --i, x--) {
      assert(i == int(*x));
    }

    // Post-decrement.
    x = size - 1;
    for (int i = size - 1; i >= 0; --i) {
      assert(i == int(*x--));
    }

  }
  // Random access iterator.
  {
    // Indexing.
    IntIterator<> x(0);
    for (std::size_t i = 0; i != size; ++i) {
      assert(i == x[i]);
    }

    // Positive offseting.
    for (std::size_t i = 0; i != size; ++i) {
      IntIterator<> y(0);
      y += i;
      assert(i == *y);
    }

    // Positive offseting.
    x = 0;
    for (std::size_t i = 0; i != size; ++i) {
      assert(i == *(x + i));
    }

    // Negative offseting.
    for (std::size_t i = 0; i != size; ++i) {
      IntIterator<> y(0);
      y -= -i;
      assert(i == *y);
    }

    // Negative offseting.
    x = 0;
    for (std::size_t i = 0; i != size; ++i) {
      assert(i == *(x - (-i)));
    }
  }

  return 0;
}
