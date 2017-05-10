// -*- C++ -*-

#include "stlib/geom/orq/BallQueryBalls.h"
#include "stlib/geom/orq/KDTree.h"

using namespace stlib;

template<typename _Float>
void
test()
{
  const std::size_t N = 3;
  typedef geom::Ball<_Float, N> Ball;

  // Empty.
  {
    const Ball* null = 0;
    geom::BallQueryBalls<N, _Float, geom::KDTree> bqb(null, null);
    std::vector<std::size_t> indices;
    {
      std::vector<std::size_t> indices;
      const Ball ball = {{{0, 0, 0}}, 1};
      bqb.query(std::back_inserter(indices), ball);
      assert(indices.empty());
    }
  }

  // Unit ball at the origin.
  {
    const Ball origin = {{{0, 0, 0}}, 1};
    geom::BallQueryBalls<N, _Float, geom::KDTree> bqb(&origin, &origin + 1);
    {
      std::vector<std::size_t> indices;
      const Ball ball = {{{0, 0, 0}}, 1};
      bqb.query(std::back_inserter(indices), ball);
      assert(indices.size() == 1);
      assert(indices[0] == 0);
    }
    {
      std::vector<std::size_t> indices;
      const Ball ball = {{{1, 0, 0}}, 1};
      bqb.query(std::back_inserter(indices), ball);
      assert(indices.size() == 1);
      assert(indices[0] == 0);
    }
    {
      std::vector<std::size_t> indices;
      const Ball ball = {{{3, 0, 0}}, 1};
      bqb.query(std::back_inserter(indices), ball);
      assert(indices.empty());
    }
    {
      std::vector<std::size_t> indices;
      const Ball ball = {{{1.5, 1.5, 1.5}}, 1};
      bqb.query(std::back_inserter(indices), ball);
      assert(indices.empty());
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
