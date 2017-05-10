// -*- C++ -*-

#include "stlib/geom/spatialIndexing/SpecialEuclideanCodeUniformGrid.h"
#include "stlib/container/SimpleMultiIndexRangeIterator.h"

#include <iostream>
#include <set>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

std::size_t
cardinality(const std::size_t bitsPerRotationCoordinate)
{
  typedef geom::SpecialEuclideanCodeUniformGrid<3> SEC;
  typedef SEC::BBox BBox;
  typedef SEC::Point Point;
  typedef SEC::Quaternion Quaternion;
  typedef container::SimpleMultiIndexRangeIterator<4> Iterator;
  typedef Iterator::IndexList IndexList;
  typedef std::array<double, 4> CoordList;

  const BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  const SEC sec(domain, 1., bitsPerRotationCoordinate);
  const Point translation = {{0, 0, 0}};
  std::size_t n = std::size_t(1) << (bitsPerRotationCoordinate + 4);
  IndexList extents = ext::filled_array<IndexList>(n);
  const double spacing = 2. / (n - 1);
  const CoordList lower = ext::filled_array<CoordList>(-1);
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
  for (std::size_t i = 3; i != 4; ++i) {
    std::cout << "bits = " << i << ", cardinality = " << cardinality(i)
              << '\n';
  }

  return 0;
}



