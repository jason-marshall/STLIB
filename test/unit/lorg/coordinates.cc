// -*- C++ -*-

#include "stlib/lorg/coordinates.h"

using namespace stlib;

template<typename _Integer, typename _Float>
void
test1()
{
  typedef lorg::DiscreteCoordinates<_Integer, _Float, 1> DiscreteCoordinates;
  typedef typename DiscreteCoordinates::Point Point;
  typedef typename DiscreteCoordinates::DiscretePoint DiscretePoint;

  const std::size_t Dimension = 1;
  const std::size_t MaxLevels =
    std::numeric_limits<_Integer>::digits / Dimension;

  std::vector<Point> positions;
  // With a single point, the length will be set to 1.
  positions.push_back(Point{{0}});
  {
    DiscreteCoordinates dc(positions);
    assert(dc.NumLevels == MaxLevels);
    assert(dc.discretize(Point{{0}}) ==
           (DiscretePoint{{0}}));
  }

  // Add a point to obtain a nontrivial domain.
  positions.push_back(Point{{2}});
  {
    DiscreteCoordinates dc(positions);
    assert(dc.NumLevels == MaxLevels);
    assert(dc.discretize(Point{{0}}) ==
           (DiscretePoint{{0}}));
  }
}

template<typename _Integer, typename _Float>
void
test2()
{
  const std::size_t Dimension = 2;
  typedef lorg::DiscreteCoordinates<_Integer, _Float, Dimension>
  DiscreteCoordinates;
  typedef typename DiscreteCoordinates::Point Point;
  typedef typename DiscreteCoordinates::DiscretePoint DiscretePoint;

  const std::size_t MaxLevels =
    std::numeric_limits<_Integer>::digits / Dimension;

  std::vector<Point> positions;
  // With a single point, the length will be set to 1.
  positions.push_back(Point{{0, 0}});
  {
    DiscreteCoordinates dc(positions);
    assert(dc.NumLevels == MaxLevels);
    assert(dc.discretize(Point{{0, 0}}) ==
           (DiscretePoint{{0, 0}}));
  }

  // Add a point to obtain a nontrivial domain.
  positions.push_back(Point{{1, 2}});
  {
    DiscreteCoordinates dc(positions);
    assert(dc.NumLevels == MaxLevels);
    assert(dc.discretize(Point{{0, 0}}) ==
           (DiscretePoint{{0, 0}}));
  }
}

int
main()
{
  test1<unsigned char, float>();
  test1<unsigned char, double>();
  test2<unsigned char, float>();
  test2<unsigned char, double>();

  test1<unsigned short, float>();
  test1<unsigned short, double>();
  test2<unsigned short, float>();
  test2<unsigned short, double>();

  test1<unsigned, float>();
  test1<unsigned, double>();
  test2<unsigned, float>();
  test2<unsigned, double>();

  test1<std::size_t, float>();
  test1<std::size_t, double>();
  test2<std::size_t, float>();
  test2<std::size_t, double>();

  return 0;
}
