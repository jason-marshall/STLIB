// -*- C++ -*-

// CONTINUE: What am I testing here?

#include "stlib/ads/functor/index.h"
#include "stlib/ads/functor/compose.h"
#include "stlib/ads/functor/Dereference.h"

#include "stlib/ads/array/FixedArray.h"

#include <cassert>

using namespace stlib;

int
main()
{
#if 0
  {
    ads::FixedArray<3> a(0, 1, 2), b(1, 2, 3);

    typedef ads::IndexConstObject< ads::FixedArray<3> > Index;
    Index ind;
    assert(ind(a, 0) == 0 && ind(b, 0) == 1);

    typedef std::binder2nd< Index > Index2;
    Index2 ind2(ind, 2);
    assert(ind2(a) == 2 && ind2(b) == 3);

    typedef ads::binary_compose_binary_unary < std::less<double>, Index2,
            Index2 > Comp;
    std::less<double> less_than;
    Comp comp(less_than, ind2, ind2);
    assert(comp(a, b) && !comp(b, a));
  }
  /*
  {
    ads::FixedArray<3> a(0, 1, 2), b(1, 2, 3);

    typedef ads::IndexObject< ads::FixedArray<3> > Index;
    Index ind;
    assert(ind(a,0) == 0 && ind(b,0) == 1);

    typedef std::binder2nd< Index > Index2;
    Index2 ind2(ind, 2);
    assert(ind2(a) == 2 && ind2(b) == 3);

    typedef ads::binary_compose_binary_unary< std::less<double>, Index2,
      Index2 > Comp;
    std::less<double> less_than;
    Comp comp(less_than, ind2, ind2);
    assert(comp(a, b) && !comp(b, a));
  }
  */
  {
    ads::FixedArray<3> a(0, 1, 2), b(1, 2, 3);
    const ads::FixedArray<3>* ap = &a;
    const ads::FixedArray<3>* bp = &b;

    typedef ads::Dereference< const ads::FixedArray<3>* > Deref;
    Deref deref;
    assert(deref(ap)[0] == 0 && deref(bp)[0] == 1);

    typedef ads::IndexConstObject< ads::FixedArray<3> > Index;
    Index ind;
    // CONTINUE: Switch to std::bind when c++11 is widely supported.
    typedef std::binder2nd< Index > Index2;
    Index2 ind2(ind, 2);
    typedef ads::unary_compose_unary_unary< Index2, Deref > I2D;
    I2D i2d(ind2, deref);
    assert(i2d(ap) == 2 && i2d(bp) == 3);

    typedef ads::binary_compose_binary_unary< std::less<double>, I2D, I2D >
    Comp;
    std::less<double> less_than;
    Comp comp(less_than, i2d, i2d);
    assert(comp(ap, bp) && !comp(bp, ap));
  }
#endif

  return 0;
}
