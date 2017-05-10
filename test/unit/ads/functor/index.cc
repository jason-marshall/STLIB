// -*- C++ -*-

#include "stlib/ads/functor/index.h"

#include <vector>

#include <cassert>

using namespace stlib;

int
main()
{
  {
    double x[3] = { 0, 1, 2 };
    ads::IndexIteratorFunctor<double*> ind;
    for (int i = 0; i != 3; ++i) {
      assert(ind(x, i) == i);
      assert(ads::index_iterator_functor<double*>()(x, i) == i);
    }
  }

  {
    std::vector<double> x(3);
    for (int i = 0; i != 3; ++i) {
      x[i] = i;
    }
    ads::IndexObject< std::vector<double> > ind;
    for (int i = 0; i != 3; ++i) {
      assert(ind(x, i) == i);
    }
  }

  // CONTINUE Replace binder2nd.
#if 0
  {
    double x[3] = { 0, 1, 2 };
    ads::IndexIteratorFunctor<double*> ind;
    std::binder2nd< ads::IndexIteratorFunctor<double*> > ind2(ind, 2);
    assert(ind2(x) == 2);
  }
#endif

  {
    double x[3] = { 0, 1, 2 };
    ads::IndexIterUnary<double*> ind(x);
    for (int i = 0; i != 3; ++i) {
      assert(ind(i) == i);
      assert(ads::index_iter_unary<double*>(x)(i) == i);
    }
  }
  return 0;
}
