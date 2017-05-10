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

  // Make the set of transformations.
  const std::size_t Size = 100000;
  std::vector<Quaternion> quaternions(Size);
  std::vector<Point> translations(Size);
  for (std::size_t i = 0; i != quaternions.size(); ++i) {
    // The rotation is a unit quaternion.
    quaternions[i] = Quaternion(random(), random(), random(), random());
    quaternions[i] /= abs(quaternions[i]);
    // The translation is in (0..1)^D.
    for (std::size_t j = 0; j != D; ++j) {
      translations[i][j] = random();
    }
  }

  // Make the functor.
  const BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  const SEC sec(domain, 1.);

  // Time the encoding.
  std::vector<Key> keys(Size);
  ads::Timer timer;
  timer.tic();
  for (std::size_t i = 0; i != quaternions.size(); ++i) {
    keys[i] = sec.encode(quaternions[i], translations[i]);
  }
  const double encodeTime = timer.toc();

  // Time the decoding.
  timer.tic();
  for (std::size_t i = 0; i != keys.size(); ++i) {
    sec.decode(keys[i], &quaternions[i], &translations[i]);
  }
  const double decodeTime = timer.toc();

  // Compute a meaningless results so that calculations are no optimized away.
  double meaningless = 0;
  for (std::size_t i = 0; i != quaternions.size(); ++i) {
    meaningless += norm(quaternions[i]) + stlib::ext::sum(translations[i]);
  }

  // Print the results.
  std::cout << "Encode = " << encodeTime / Size * 1e9 << " ns. "
            << "Decode = " << decodeTime / Size * 1e9 << " ns.\n"
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



