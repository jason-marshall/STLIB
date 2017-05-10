// -*- C++ -*-

#include "stlib/geom/orq/ORQ.h"

#include "stlib/ads/functor/Dereference.h"

#include <iostream>

#include <cassert>

using namespace stlib;

template<typename _Float>
void
test()
{
  typedef std::array<_Float, 3> Point;
  typedef typename std::vector<Point>::const_iterator Record;
  typedef geom::Orq<3, ads::Dereference<Record> > Orq;

  {
    //
    // Constructors.
    //
    std::cout << "Orq() = " << '\n'
              << Orq() << '\n';
    Orq x;
    Orq y(x);
    Orq z;
    z = x;

    //
    // Accessors.
    //
    assert(x.size() == 0);
    assert(x.empty());
    assert(y.empty());
  }
}

int
main()
{
  test<float>();
  test<double>();

  return 0;
}
