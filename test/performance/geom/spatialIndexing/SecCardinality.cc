// -*- C++ -*-

#include "stlib/geom/spatialIndexing/SpecialEuclideanCode.h"
#include "stlib/container/SimpleMultiIndexRangeIterator.h"

#include <iostream>
#include <set>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

template<std::size_t _SubdivisionLevels>
std::size_t
cardinality()
{
  typedef geom::SpecialEuclideanCode<3, _SubdivisionLevels> SEC;
  typedef typename SEC::BBox BBox;
  typedef typename SEC::Point Point;
  typedef typename SEC::Quaternion Quaternion;
  typedef container::SimpleMultiIndexRangeIterator<4> Iterator;
  typedef Iterator::IndexList IndexList;
  typedef std::array<double, 4> CoordList;

  const BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  const SEC sec(domain, 1.);
  const Point translation = {{0, 0, 0}};
  const CoordList lower = ext::filled_array<CoordList>(-1);
  const std::size_t n = 64;
  const IndexList extents = ext::filled_array<IndexList>(n);
  const double spacing = 2. / (n - 1);
  CoordList x;
  Quaternion q;
  std::set<std::size_t> keys;
  const Iterator end = Iterator::end(extents);
  for (Iterator i = Iterator::begin(extents); i != end; ++i) {
    x = lower + spacing * ext::convert_array<double>(*i);
    stlib::ext::normalize(&x);
    q = Quaternion(x[0], x[1], x[2], x[3]);
    keys.insert(sec.encode(q, translation));
  }
  return keys.size();
}


int
main()
{
  std::cout << "levels = 1, cardinality = " << cardinality<1>() << '\n'
            << "levels = 2, cardinality = " << cardinality<2>() << '\n'
            << "levels = 3, cardinality = " << cardinality<3>() << '\n';

  return 0;
}



