// -*- C++ -*-

#include "stlib/levelSet/GridUniform.h"

#include "stlib/numerical/equality.h"
#include "stlib/geom/kernel/Ball.h"

#include <iostream>
#include <sstream>

using namespace stlib;

template<std::size_t D>
void
test()
{
  using numerical::areEqual;

  typedef double Number;
  typedef std::array<Number, D> Point;
  typedef geom::BBox<Number, D> BBox;
  typedef levelSet::GridUniform<Number, D> GridUniform;

  std::cout << "\nDimension = " << D << '\n';

  {
    // Spacing.
    const Point lower = ext::filled_array<Point>(0);
    const Point upper = ext::filled_array<Point>(1);
    const BBox domain = {lower, upper};
    {
      GridUniform x(domain, 2);
      assert(areEqual(x.spacing, 1));
      assert(areEqual(x.domain().lower, domain.lower));
      assert(areEqual(x.domain().upper, domain.upper));
    }
    {
      GridUniform x(domain, 1.001);
      assert(areEqual(x.spacing, 1));
    }
    {
      GridUniform x(domain, 0.5001);
      assert(areEqual(x.spacing, 0.5));
    }
    {
      GridUniform x(domain, 0.1001);
      assert(areEqual(x.spacing, 0.1));
    }

    // +=
    {
      GridUniform x(domain, 1.001);
      x[0] = 0;
      x += Number(1);
      assert(x[0] == 1);
    }
  }
}


int
main()
{
  test<1>();
  test<2>();
  test<3>();

  // writeVtkXml()
  {
    // 2-D
    const std::size_t Dimension = 2;
    typedef float Number;
    typedef levelSet::GridUniform<Number, Dimension> GridUniform;
    typedef GridUniform::BBox BBox;
    typedef GridUniform::Point Point;
    typedef GridUniform::IndexList IndexList;
    typedef geom::Ball<Number, Dimension> Ball;
    typedef container::SimpleMultiIndexRangeIterator<Dimension> Iterator;

    const BBox domain = {{{0, 0}}, {{1, 1}}};
    GridUniform grid(domain, 1. / (10 - 1) * 1.01);
    assert(grid.extents() == ext::filled_array<IndexList>(10));
    // Compute the distance to a ball.
    const Ball ball = {{{0.5, 0.5}}, 0.5};
    const Iterator iEnd = Iterator::end(grid.extents());
    for (Iterator i = Iterator::begin(grid.extents()); i != iEnd; ++i) {
      const Point x = grid.indexToLocation(*i);
      grid(*i) = stlib::ext::euclideanDistance(ball.center, x) - ball.radius;
    }
    std::ostringstream out;
    writeVtkXml(grid, out);
  }
  {
    // 3-D
    const std::size_t Dimension = 3;
    typedef float Number;
    typedef levelSet::GridUniform<Number, Dimension> GridUniform;
    typedef GridUniform::BBox BBox;
    typedef GridUniform::Point Point;
    typedef GridUniform::IndexList IndexList;
    typedef geom::Ball<Number, Dimension> Ball;
    typedef container::SimpleMultiIndexRangeIterator<Dimension> Iterator;

    const BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
    GridUniform grid(domain, 1. / (10 - 1) * 1.01);
    assert(grid.extents() == ext::filled_array<IndexList>(10));
    // Compute the distance to a ball.
    const Ball ball = {{{0.5, 0.5, 0.5}}, 0.5};
    const Iterator iEnd = Iterator::end(grid.extents());
    for (Iterator i = Iterator::begin(grid.extents()); i != iEnd; ++i) {
      const Point x = grid.indexToLocation(*i);
      grid(*i) = stlib::ext::euclideanDistance(ball.center, x) - ball.radius;
    }
    std::ostringstream out;
    writeVtkXml(grid, out);
  }

  return 0;
}
