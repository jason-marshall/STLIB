// -*- C++ -*-

#include "stlib/simd/functions.h"
#include "stlib/simd/align.h"
#include "stlib/simd/constants.h"

#include <limits>
// CONTINUE REMOVE
#include <iostream>

#include <cassert>


using namespace stlib;

template<typename _Float>
void
test()
{
  typedef float Float;
  typedef simd::Vector<Float>::Type Vector;
  const std::size_t VectorSize = simd::Vector<Float>::Size;

#if 0
  // NaN has a range of possible representations. It is not necessarily 
  // bitwise ones.
  {
    const Vector a = simd::set1(std::numeric_limits<Float>::quiet_NaN());
    ALIGN_SIMD Float v[VectorSize];
    simd::store(v, a);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(v[i] != v[i]);
    }
  }
#endif

  // setzero()
  {
    Vector a = simd::setzero<Float>();
    ALIGN_SIMD Float av[VectorSize];
    simd::store(av, a);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(av[i] == 0);
    }
  }

  // setBits()
  {
    Vector a = simd::setBits<Float>();
    ALIGN_SIMD Float av[VectorSize];
    simd::store(av, a);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(std::isnan(av[i]));
    }
  }

  // set1()
  {
    const Float Value = 23;
    Vector a = simd::set1(Value);
    ALIGN_SIMD Float b[VectorSize];
    simd::store(b, a);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(b[i] == Value);
    }
  }

  // load()
  {
    ALIGN_SIMD Float a[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      a[i] = i + 1;
    }
    Vector v = simd::load(a);
    ALIGN_SIMD Float b[VectorSize];
    simd::store(b, v);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(b[i] == i + 1);
    }
  }

  // front()
  {
    ALIGN_SIMD Float a[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      a[i] = i + 1;
    }
    Vector v = simd::load(a);
    assert(simd::front(v) == 1);
  }

  // sum()
  {
    ALIGN_SIMD Float a[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      a[i] = i;
    }
    Vector v = simd::load(a);
    assert(simd::sum(v) == (VectorSize - 1) * VectorSize / 2);
  }

  // bitwise and
  {
    const Vector f = simd::set1(Float(0));
    const Vector t = simd::set1(std::numeric_limits<Float>::quiet_NaN());
    ALIGN_SIMD Float v[VectorSize];
    simd::store(v, simd::bitwiseAnd(t, t));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(v[i] != v[i]);
    }
    simd::store(v, simd::bitwiseAnd(t, f));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(v[i] == 0);
    }
    simd::store(v, simd::bitwiseAnd(f, t));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(v[i] == 0);
    }
    simd::store(v, simd::bitwiseAnd(f, f));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(v[i] == 0);
    }
  }

  // bitwise or
  {
    const Vector f = simd::set1(Float(0));
    const Vector t = simd::set1(std::numeric_limits<Float>::quiet_NaN());
    ALIGN_SIMD Float v[VectorSize];
    simd::store(v, simd::bitwiseOr(t, t));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(v[i] != v[i]);
    }
    simd::store(v, simd::bitwiseOr(t, f));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(v[i] != v[i]);
    }
    simd::store(v, simd::bitwiseOr(f, t));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(v[i] != v[i]);
    }
    simd::store(v, simd::bitwiseOr(f, f));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(v[i] == 0);
    }
  }

  // min()
  {
    ALIGN_SIMD Float av[VectorSize];
    ALIGN_SIMD Float bv[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      av[i] = i;
      bv[i] = i + 1;
    }
    const Vector a = simd::load(av);
    const Vector b = simd::load(bv);
    {
      const Vector c = simd::min(a, b);
      ALIGN_SIMD Float cv[VectorSize];
      simd::store(cv, c);
      for (std::size_t i = 0; i != VectorSize; ++i) {
        assert(cv[i] == av[i]);
      }
    }
    {
      const Vector c = simd::min(b, a);
      ALIGN_SIMD Float cv[VectorSize];
      simd::store(cv, c);
      for (std::size_t i = 0; i != VectorSize; ++i) {
        assert(cv[i] == av[i]);
      }
    }
  }

  // min() with NaN's
  // This is really wierd. min(a, b) may not be the same as min(b, a).
  {
    ALIGN_SIMD Float av[VectorSize];
    ALIGN_SIMD Float bv[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      av[i] = i;
      bv[i] = std::numeric_limits<Float>::quiet_NaN();
    }
    const Vector a = simd::load(av);
    const Vector b = simd::load(bv);
    {
      const Vector c = simd::min(a, b);
      ALIGN_SIMD Float cv[VectorSize];
      simd::store(cv, c);
      for (std::size_t i = 0; i != VectorSize; ++i) {
        assert(cv[i] != cv[i]);
      }
    }
    {
      const Vector c = simd::min(b, a);
      ALIGN_SIMD Float cv[VectorSize];
      simd::store(cv, c);
      for (std::size_t i = 0; i != VectorSize; ++i) {
        assert(cv[i] == av[i]);
      }
    }
  }

  // max()
  {
    ALIGN_SIMD Float av[VectorSize];
    ALIGN_SIMD Float bv[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      av[i] = i;
      bv[i] = i + 1;
    }
    const Vector a = simd::load(av);
    const Vector b = simd::load(bv);
    {
      const Vector c = simd::max(a, b);
      ALIGN_SIMD Float cv[VectorSize];
      simd::store(cv, c);
      for (std::size_t i = 0; i != VectorSize; ++i) {
        assert(cv[i] == bv[i]);
      }
    }
    {
      const Vector c = simd::max(b, a);
      ALIGN_SIMD Float cv[VectorSize];
      simd::store(cv, c);
      for (std::size_t i = 0; i != VectorSize; ++i) {
        assert(cv[i] == bv[i]);
      }
    }
  }

  // equal()
  {
    ALIGN_SIMD Float a[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      a[i] = i;
    }
    const Vector x = simd::load(a);
    Vector y = simd::equal(x, x);
    const Float* d = reinterpret_cast<const Float*>(&y);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] != d[i]);
    }
    y = simd::equal(x, x + simd::set1(Float(1)));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] == 0);
    }
  }

  // min() - horizontal
  {
    ALIGN_SIMD Float av[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      av[i] = i;
    }
    assert(simd::min(simd::load(av)) == 0);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      av[i] = VectorSize - i - 1;
    }
    assert(simd::min(simd::load(av)) == 0);
  }

  // max() - horizontal
  {
    ALIGN_SIMD Float av[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      av[i] = i;
    }
    assert(simd::max(simd::load(av)) == VectorSize - 1);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      av[i] = VectorSize - i - 1;
    }
    assert(simd::max(simd::load(av)) == VectorSize - 1);
  }

  // notEqual()
  {
    ALIGN_SIMD Float a[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      a[i] = i;
    }
    const Vector x = simd::load(a);
    Vector y = simd::notEqual(x, x);
    const Float* d = reinterpret_cast<const Float*>(&y);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] == 0);
    }
    y = simd::notEqual(x, x + simd::set1(Float(1)));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] != d[i]);
    }
  }

  // less()
  {
    ALIGN_SIMD Float a[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      a[i] = i;
    }
    const Vector x = simd::load(a);
    Vector y = simd::less(x, x);
    const Float* d = reinterpret_cast<const Float*>(&y);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] == 0);
    }
    y = simd::less(x, x + simd::set1(Float(1)));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] != d[i]);
    }
    y = simd::less(x, x - simd::set1(Float(1)));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] == 0);
    }
  }

  // lessEqual()
  {
    ALIGN_SIMD Float a[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      a[i] = i;
    }
    const Vector x = simd::load(a);
    Vector y = simd::lessEqual(x, x);
    const Float* d = reinterpret_cast<const Float*>(&y);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] != d[i]);
    }
    y = simd::lessEqual(x, x + simd::set1(Float(1)));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] != d[i]);
    }
    y = simd::lessEqual(x, x - simd::set1(Float(1)));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] == 0);
    }
  }

  // greater()
  {
    ALIGN_SIMD Float a[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      a[i] = i;
    }
    const Vector x = simd::load(a);
    Vector y = simd::greater(x, x);
    const Float* d = reinterpret_cast<const Float*>(&y);
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] == 0);
    }
    y = simd::greater(x, x + simd::set1(Float(1)));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] == 0);
    }
    y = simd::greater(x, x - simd::set1(Float(1)));
    for (std::size_t i = 0; i != VectorSize; ++i) {
      assert(d[i] != d[i]);
    }
  }

  // moveMask()
  {
    Vector x = simd::setzero<Float>();
    assert(simd::moveMask(x) == 0);
    x = -x;
    assert(simd::moveMask(x) == (1 << VectorSize) - 1);
  }

  // abs()
  {
    ALIGN_SIMD Float av[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      av[i] = -Float(i) - 1;
    }
    const Vector a = simd::load(av);
    {
      const Vector c = simd::abs(a);
      ALIGN_SIMD Float cv[VectorSize];
      simd::store(cv, c);
      for (std::size_t i = 0; i != VectorSize; ++i) {
        assert(cv[i] == std::abs(av[i]));
      }
    }
  }
  // sqrt()
  {
    Float const Tol = 10 * std::numeric_limits<Float>::epsilon();
    ALIGN_SIMD Float av[VectorSize];
    for (std::size_t i = 0; i != VectorSize; ++i) {
      av[i] = i;
    }
    const Vector a = simd::load(av);
    {
      const Vector c = simd::sqrt(a);
      ALIGN_SIMD Float cv[VectorSize];
      simd::store(cv, c);
      for (std::size_t i = 0; i != VectorSize; ++i) {
        assert(std::abs(cv[i] - std::sqrt(av[i])) < Tol);
      }
    }
  }
}


int
main()
{
  test<float>();
  test<double>();

  return 0;
}
