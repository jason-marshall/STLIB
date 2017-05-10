// -*- C++ -*-

#include "stlib/geom/spatialIndexing/SpecialEuclideanCode.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/ads/timer.h"

#include <iostream>


using namespace stlib;

template<std::size_t _SubdivisionLevels>
void
time()
{
  const std::size_t D = 3;
  typedef geom::SpecialEuclideanCode<D, _SubdivisionLevels> SEC;
  typedef typename SEC::Key Key;
  typedef typename SEC::BBox BBox;
  typedef typename SEC::Point Point;
  typedef typename SEC::Quaternion Quaternion;

  // Make the random number generator.
  typedef numerical::ContinuousUniformGeneratorOpen<> Continuous;
  typedef Continuous::DiscreteUniformGenerator Discrete;
  Discrete discrete;
  Continuous random(&discrete);

  // Make the functor.
  const BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  const SEC sec(domain, 0.01);

  // Make the set of transformations.
  std::vector<Key> keys(10000);
  for (std::size_t i = 0; i != keys.size(); ++i) {
    // The rotation is a unit quaternion.
    Quaternion q = Quaternion(random(), random(), random(), random());
    q /= abs(q);
    // The translation is in (0..1)^D.
    const Point t = {{random(), random(), random()}};
    keys[i] = sec.encode(q, t);
  }

  // Time the neighbor search.
  std::vector<Key> neighbors;
  // Compute a meaningless results so that calculations are not optimized away.
  std::size_t meaningless = 0;
  ads::Timer timer;
  timer.tic();
  for (std::size_t i = 0; i != keys.size(); ++i) {
    sec.neighbors(keys[i], &neighbors);
    meaningless += neighbors.size();
  }
  const double time = timer.toc();

  // Print the results.
  std::cout << "Time = " << time / keys.size() * 1e6 << " microseconds. "
            << "Meaningless result = " << meaningless << "\n\n";
}


int
main()
{
  time<1>();
  time<2>();
  time<3>();

  return 0;
}



