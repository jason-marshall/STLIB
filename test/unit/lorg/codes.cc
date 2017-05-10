// -*- C++ -*-

#include "stlib/lorg/codes.h"

using namespace stlib;

template<typename _Integer, typename _Float>
void
test1()
{
  const std::size_t Dimension = 1;
  typedef lorg::Morton<_Integer, _Float, Dimension> Morton;
  typedef typename Morton::Point Point;

  const std::size_t MaxLevels =
    std::numeric_limits<_Integer>::digits / Dimension;

  std::vector<Point> positions;
  // With a single point, the length will be set to 1.
  positions.push_back(Point{{0}});
  {
    Morton morton(positions);
    assert(morton.NumLevels == MaxLevels);
    assert(morton.code(Point{{0}}) == 0);
  }
}

template<typename _Integer, typename _Float>
void
test2()
{
  const std::size_t Dimension = 2;
  typedef lorg::Morton<_Integer, _Float, Dimension> Morton;
  typedef typename Morton::Point Point;

  const std::size_t MaxLevels =
    std::numeric_limits<_Integer>::digits / Dimension;

  std::vector<Point> positions;
  // With a single point, the length will be set to 1.
  positions.push_back(Point{{0, 0}});
  {
    Morton morton(positions);
    assert(morton.NumLevels == MaxLevels);
    assert(morton.code(Point{{0, 0}}) == 0);
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
