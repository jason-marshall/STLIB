// -*- C++ -*-

#include "stlib/numerical/equality.h"

#include <cassert>

using namespace stlib;
using numerical::areEqual;
using numerical::areSequencesEqual;
using numerical::areEqualAbs;
using numerical::areSequencesEqualAbs;
using numerical::isSmall;

template<typename _T>
void
testAreEqual()
{
  const _T Eps = std::numeric_limits<_T>::epsilon();
  {
    _T x = 1;
    _T y = 1 + Eps;
    _T t = 0.5;
    assert(areEqual(x, y));
    assert(! areEqual(x, y, t));
    y = 1 + 4 * Eps;
    assert(! areEqual(x, y));
    x = Eps;
    y = x * (1 + Eps);
    assert(areEqual(x, y));
    y = 4 * Eps;
    assert(! areEqual(x, y));
    x = 0;
    y = 3 * Eps;
    assert(! areEqual(x, y));

    x = std::numeric_limits<_T>::max();
    y = x * (1 - Eps);
    assert(areEqual(x, y));
    y = x * (1 - 4 * Eps);
    assert(! areEqual(x, y));

    x = std::numeric_limits<_T>::min();
    y = x * (1 + Eps);
    assert(areEqual(x, y));
    y = x + 3 * Eps;
    assert(! areEqual(x, y));
  }
  {
    const _T dataX[] = {0, 1};
    const _T* x = dataX;
    assert(areSequencesEqual(x, x + 2, x));
    const _T dataY[] = {1, 1};
    const _T* y = dataY;
    assert(! areSequencesEqual(x, x + 2, y));
  }
  {
    const std::array<_T, 2> x = {{0, 1}};
    assert(areEqual(x, x));
    const std::array<_T, 2> y = {{1, 1}};
    assert(! areEqual(x, y));
  }
}

template<typename _T>
void
testAreEqualAbs()
{
  const _T Eps = std::numeric_limits<_T>::epsilon();
  {
    _T x = 1;
    _T y = 1 + Eps;
    const _T s = 1;
    const _T t = 0.5;
    assert(areEqualAbs(x, y));
    assert(! areEqualAbs(x, y, s, t));
    y = 1 + 4 * Eps;
    assert(! areEqualAbs(x, y));
    x = Eps;
    y = x * (1 + Eps);
    assert(areEqualAbs(x, y));

    x = std::numeric_limits<_T>::max();
    y = x * (1 - Eps);
    assert(! areEqualAbs(x, y));
    assert(areEqualAbs(x, y, std::numeric_limits<_T>::max()));

    x = std::numeric_limits<_T>::min();
    y = x * (1 + Eps);
    assert(areEqualAbs(x, y));
  }
  {
    const _T dataX[] = {0, 1};
    const _T* x = dataX;
    assert(areSequencesEqualAbs(x, x + 2, x));
    const _T dataY[] = {1, 1};
    const _T* y = dataY;
    assert(! areSequencesEqualAbs(x, x + 2, y));
  }
  {
    const std::array<_T, 2> x = {{0, 1}};
    assert(areEqualAbs(x, x));
    const std::array<_T, 2> y = {{1, 1}};
    assert(! areEqualAbs(x, y));
  }
}

template<typename _T>
void
testIsSmall()
{
  const _T Eps = std::numeric_limits<_T>::epsilon();
  _T x = Eps;
  assert(isSmall(x));
  _T t = 0.5;
  assert(! isSmall(x, t));
  x = 2 * Eps;
  assert(! isSmall(x));
}

int
main()
{
  testAreEqual<double>();
  testAreEqual<float>();
  testAreEqualAbs<double>();
  testAreEqualAbs<float>();
  testIsSmall<double>();
  testIsSmall<float>();

  return 0;
}
