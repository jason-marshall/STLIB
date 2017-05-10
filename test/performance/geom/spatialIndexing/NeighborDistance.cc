// -*- C++ -*-

#include "stlib/geom/spatialIndexing/SpecialEuclideanCode.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/ads/timer.h"

#include <iostream>


using namespace stlib;

// Euclidean distance for unit quaternions.
double
euclideanDistance(const boost::math::quaternion<double>& a,
                  const boost::math::quaternion<double>& b)
{
  return std::min(abs(a - b), abs(a + b));
}


void
translation(const double spacing)
{
  const std::size_t D = 3;
  typedef geom::SpecialEuclideanCode<D, 1> SEC;
  typedef SEC::Key Key;
  typedef SEC::BBox BBox;
  typedef SEC::Point Point;
  typedef SEC::Quaternion Quaternion;

  // Make the random number generator.
  typedef numerical::ContinuousUniformGeneratorOpen<> Continuous;
  typedef Continuous::DiscreteUniformGenerator Discrete;
  Discrete discrete;
  Continuous random(&discrete);

  // Make the functor.
  const BBox domain = {{{0, 0, 0}}, {{1, 1, 1}}};
  const SEC sec(domain, spacing);

  // Make the set of transformations.
  const std::size_t Size = 1000;
  std::vector<Point> translations(Size);
  std::vector<Key> keys(Size);
  const Quaternion quaternion(1, 0, 0 , 0);
  for (std::size_t i = 0; i != keys.size(); ++i) {
    // The translation is in (0..1)^D.
    for (std::size_t j = 0; j != D; ++j) {
      translations[i][j] = random();
    }
    keys[i] = sec.encode(quaternion, translations[i]);
  }

  double maxClose = 0;
  double minFar = std::numeric_limits<double>::infinity();
  std::vector<Key> neighbors;
  std::set<Key> neighborSet;
  for (std::size_t i = 0; i != keys.size(); ++i) {
    sec.neighbors(keys[i], &neighbors);
    neighborSet.clear();
    neighborSet.insert(neighbors.begin(), neighbors.end());
    for (std::size_t j = 0; j != keys.size(); ++j) {
      const double d =
        stlib::ext::euclideanDistance(translations[i], translations[j]);
      // If this is a neighbor.
      if (neighborSet.count(keys[j])) {
        maxClose = std::max(maxClose, d);
      }
      else {
        minFar = std::min(minFar, d);
      }
    }
  }

  assert(maxClose < 1.01 * 2 * std::sqrt(3.) * spacing);
  assert(minFar > 0.99 * spacing);

  // Print the results.
  std::cout << "Number of translations = " << keys.size() << '\n'
            << "spacing = " << spacing << '\n'
            << "maxClose = " << maxClose << '\n'
            << "minFar = " << minFar << "\n\n";
}


template<std::size_t _SubdivisionLevels>
void
rotation()
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
  const double Spacing = 0.01;
  const SEC sec(domain, Spacing);

  // Make the set of transformations.
  const std::size_t Size = 20000;
  std::vector<Quaternion> quaternions(Size);
  std::vector<Key> keys(Size);
  const Point translation = {{0, 0, 0}};
  for (std::size_t i = 0; i != keys.size(); ++i) {
    // The rotation is a unit quaternion.
    quaternions[i] = Quaternion(random(), random(), random(), random());
    quaternions[i] /= abs(quaternions[i]);
    keys[i] = sec.encode(quaternions[i], translation);
  }

  double maxClose = 0;
  double minFar = std::numeric_limits<double>::infinity();
  std::vector<Key> neighbors;
  std::set<Key> neighborSet;
  for (std::size_t i = 0; i != keys.size(); ++i) {
    sec.neighbors(keys[i], &neighbors);
    neighborSet.clear();
    neighborSet.insert(neighbors.begin(), neighbors.end());
    for (std::size_t j = 0; j != keys.size(); ++j) {
      const double d = euclideanDistance(quaternions[i], quaternions[j]);
      // If this is a neighbor.
      if (neighborSet.count(keys[j])) {
        maxClose = std::max(maxClose, d);
      }
      else {
        minFar = std::min(minFar, d);
      }
    }
  }

  // Print the results.
  std::cout << "Number of translations = " << keys.size() << '\n'
            << "Levels of subdivision = " << _SubdivisionLevels << '\n'
            << "maxClose = " << maxClose << '\n'
            << "minFar = " << minFar << "\n\n";
}


int
main()
{
  std::cout << "-----------------------------------------------------------\n"
            << "Translations:\n\n";
  translation(1.);
  translation(0.1);
  translation(0.01);
  std::cout << "-----------------------------------------------------------\n"
            << "Rotations:\n\n";
  rotation<1>();
  rotation<2>();
  rotation<3>();
  rotation<4>();

  return 0;
}



