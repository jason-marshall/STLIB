// -*- C++ -*-

#include "stlib/stochastic/HistogramReference.h"

using namespace stlib;

int
main()
{
  typedef stochastic::HistogramReference HistogramReference;

  {
    // Empty.
    const std::size_t Size = 32;
    double bins[Size];
    HistogramReference x(Size, bins);
    assert(x.size() == Size);
    assert(x.computeSum() == 0);
    assert(x.getLowerBound() == 0);
    assert(x.getWidth() == 1);
    assert(x.getUpperBound() == Size);
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(x[i] == 0);
    }
  }

  {
    // Insert within range.
    const std::size_t Size = 32;
    double bins[Size];
    HistogramReference x(Size, bins);
    for (std::size_t i = 0; i != Size; ++i) {
      x.accumulate(i, 1);
    }
    assert(x.size() == Size);
    assert(x.computeSum() == Size);
    assert(x.getLowerBound() == 0);
    assert(x.getWidth() == 1);
    assert(x.getUpperBound() == Size);
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(x[i] == 1);
    }
    // Copy constructor.
    {
      HistogramReference y = x;
      assert(y.size() == x.size());
      assert(y.computeSum() == x.computeSum());
      assert(y.getLowerBound() == x.getLowerBound());
      assert(y.getWidth() == x.getWidth());
      assert(y.getUpperBound() == x.getUpperBound());
      for (std::size_t i = 0; i != x.size(); ++i) {
        assert(y[i] == x[i]);
      }
    }
    // Assignment operator.
    {
      HistogramReference y;
      y = x;
      assert(y.size() == x.size());
      assert(y.computeSum() == x.computeSum());
      assert(y.getLowerBound() == x.getLowerBound());
      assert(y.getWidth() == x.getWidth());
      assert(y.getUpperBound() == x.getUpperBound());
      for (std::size_t i = 0; i != x.size(); ++i) {
        assert(y[i] == x[i]);
      }
    }
  }

  {
    // Insert above range.
    const std::size_t Size = 32;
    double bins[Size];
    HistogramReference x(Size, bins);
    for (std::size_t i = 0; i != 2 * Size; ++i) {
      x.accumulate(i, 1);
    }
    assert(x.size() == Size);
    assert(x.computeSum() == 2 * Size);
    assert(x.getLowerBound() == 0);
    assert(x.getWidth() == 2);
    assert(x.getUpperBound() == 2 * Size);
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(x[i] == 2);
    }
  }

  {
    // Insert above range.
    const std::size_t Size = 32;
    double bins[Size];
    HistogramReference x(Size, bins);
    x.accumulate(100, 1);
    assert(x.size() == Size);
    assert(x.computeSum() == 1);
    assert(x.getLowerBound() == 100);
    assert(x.getWidth() == 1);
    assert(x.getUpperBound() == 100 + Size);
    assert(x[0] == 1);
    for (std::size_t i = 1; i != x.size(); ++i) {
      assert(x[i] == 0);
    }
  }

  {
    // Insert below range.
    const std::size_t Size = 32;
    double bins[Size];
    HistogramReference x(Size, bins);
    for (std::size_t i = 2 * Size; i != 0; --i) {
      x.accumulate(i - 1, 1);
    }
    assert(x.size() == Size);
    assert(x.computeSum() == 2 * Size);
    assert(x.getLowerBound() == 0);
    assert(x.getWidth() == 2);
    assert(x.getUpperBound() == 2 * Size);
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(x[i] == 2);
    }

    x.reset();
    assert(x.size() == Size);
    assert(x.computeSum() == 0);
    assert(x.getLowerBound() == 0);
    assert(x.getWidth() == 1);
    assert(x.getUpperBound() == Size);
    for (std::size_t i = 0; i != x.size(); ++i) {
      assert(x[i] == 0);
    }
  }

  return 0;
}
