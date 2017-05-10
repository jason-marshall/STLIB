// -*- C++ -*-

#include "stlib/geom/orq/IntersectionQuerySimplexBalls.h"
#include "stlib/geom/orq/KDTree.h"

using namespace stlib;

template<typename _Float>
void
test()
{
  const std::size_t N = 3;
  typedef geom::IntersectionQuerySimplexBalls<N, _Float, geom::KDTree>
  IntersectionQuery;
  typedef typename IntersectionQuery::Ball Ball;
  typedef typename IntersectionQuery::Simplex Simplex;

  // Empty.
  {
    const Ball* null = 0;
    IntersectionQuery iq(null, null);
    std::vector<std::size_t> indices;
    {
      std::vector<std::size_t> indices;
      const Simplex simplex = {{{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}},
          {{0, 0, 1}}
        }
      };
      iq.query(std::back_inserter(indices), simplex);
      assert(indices.empty());
    }
  }

  // Unit ball at the origin.
  {
    const Ball origin = {{{0, 0, 0}}, 1};
    IntersectionQuery iq(&origin, &origin + 1);
    {
      std::vector<std::size_t> indices;
      const Simplex simplex = {{{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}},
          {{0, 0, 1}}
        }
      };
      iq.query(std::back_inserter(indices), simplex);
      assert(indices.size() == 1);
      assert(indices[0] == 0);
    }
    {
      std::vector<std::size_t> indices;
      const Simplex simplex = {{{{0.99, 0, 0}}, {{1.99, 0, 0}},
          {{0.99, 1, 0}}, {{0.99, 0, 1}}
        }
      };
      iq.query(std::back_inserter(indices), simplex);
      assert(indices.size() == 1);
      assert(indices[0] == 0);
    }
    {
      std::vector<std::size_t> indices;
      const Simplex simplex = {{{{1.01, 0, 0}}, {{2.01, 0, 0}},
          {{1.01, 1, 0}}, {{1.01, 0, 1}}
        }
      };
      iq.query(std::back_inserter(indices), simplex);
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
