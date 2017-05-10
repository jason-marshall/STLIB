// -*- C++ -*-

#include "stlib/geom/orq/CellArrayStaticPeriodic.h"
#include "stlib/ads/functor/Dereference.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <set>

using namespace stlib;

template<std::size_t _N, typename _Float>
void
test()
{
  typedef std::array<_Float, _N> Value;
  typedef std::vector<Value> ValueContainer;
  typedef typename ValueContainer::const_iterator Record;
  typedef geom::CellArrayStaticPeriodic<_N, ads::Dereference<Record> > Casp;
  typedef typename Casp::Point Point;
  typedef typename Casp::BBox BBox;
  typedef typename Casp::Ball Ball;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  const _Float Eps = 10 * std::numeric_limits<_Float>::epsilon();
  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  // Make a vector of random points.
  ValueContainer values(100);
  for (std::size_t i = 0; i != values.size(); ++i) {
    for (std::size_t j = 0; j != _N; ++j) {
      values[i][j] = random();
    }
  }

  const BBox Domain = {ext::filled_array<Point>(0),
                       ext::filled_array<Point>(1)
                      };
  // Structure for computing periodic distance.
  geom::DistancePeriodic<_Float, _N> dp(Domain);
  // Make the neighbor query data structure.
  Casp casp(Domain, values.begin(), values.end());

  // For each record.
  for (std::size_t i = 0; i != values.size(); ++i) {
    // Make a query ball.
    const Ball ball = {values[i], _Float(random())};
    std::vector<Record> neighbors;
    casp.neighborQuery(std::back_inserter(neighbors), ball);
    // Check that the reported neighbors are actually close.
    for (std::size_t j = 0; j != neighbors.size(); ++j) {
      assert(dp.distance(values[i], *neighbors[j]) <= ball.radius + Eps);
    }
    // Check that the points that are not neighbors are not close.
    std::set<std::size_t> neighborIndices;
    for (std::size_t j = 0; j != neighbors.size(); ++j) {
      neighborIndices.insert(std::distance(Record(values.begin()),
                                           neighbors[j]));
    }
    for (std::size_t j = 0; j != values.size(); ++j) {
      if (neighborIndices.count(j)) {
        continue;
      }
      assert(dp.distance(values[i], values[j]) > ball.radius - Eps);
    }
  }
}

int
main()
{
  test<1, float>();
  test<1, double>();

  return 0;
}
