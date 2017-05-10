// -*- C++ -*-

#include "stlib/particle/order.h"
#include "stlib/particle/traits.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/ads/functor/Identity.h"

using namespace stlib;

template<typename _Traits>
class MortonOrder :
  public particle::MortonOrder<_Traits>
{
public:
  typedef typename _Traits::Float Float;
  typedef particle::MortonOrder<_Traits> Base;

  MortonOrder(const geom::BBox<Float, _Traits::Dimension>& domain,
              const Float interactionDistance, const Float padding) :
    Base(domain, interactionDistance, padding)
  {
  }

  using Base::morton;
  using Base::_cellCodes;
  using Base::index;
};

template<typename _Float, std::size_t _Dimension>
struct SetPosition {
  void
  operator()(std::array<_Float, _Dimension>* particle,
             const std::array<_Float, _Dimension>& point) const
  {
    *particle = point;
  }
};


template<std::size_t _Dimension, bool _Periodic, typename _Float>
void
testPadding(const _Float interactionDistance)
{
  typedef std::array<_Float, _Dimension> Particle;
  typedef particle::Traits<Particle, ads::Identity<Particle>,
          SetPosition<_Float, _Dimension>, _Periodic,
          _Dimension, _Float> Traits;
  typedef particle::MortonOrder<Traits> MortonOrder;
  typedef typename MortonOrder::Point Point;

  // Make the data structure for ordering the particles.
  const geom::BBox<_Float, _Dimension> Domain = {ext::filled_array<Point>(0),
                                                 ext::filled_array<Point>(1)
                                                };
  MortonOrder mortonOrder(Domain, interactionDistance);

  assert(0 < mortonOrder.padding() &&
         mortonOrder.padding() <= mortonOrder.interactionDistance());
}


template<std::size_t _Dimension, bool _Periodic, typename _Float>
void
test(const _Float interactionDistance, const _Float padding)
{
  typedef particle::IntegerTypes::Code Code;
  typedef std::array<_Float, _Dimension> Particle;
  typedef particle::Traits<Particle, ads::Identity<Particle>,
          SetPosition<_Float, _Dimension>, _Periodic,
          _Dimension, _Float> Traits;
  typedef MortonOrder<Traits> MortonOrder;

  typedef typename MortonOrder::Point Point;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  // Make the data structure for ordering the particles.
  const geom::BBox<_Float, _Dimension> Domain = {ext::filled_array<Point>(0),
                                                 ext::filled_array<Point>(1)
                                                };
  MortonOrder mortonOrder(Domain, interactionDistance, padding);

  {
    // The random number generator.
    ContinuousUniformGenerator::DiscreteUniformGenerator generator;
    ContinuousUniformGenerator random(&generator);

    // Make a vector of particles with random positions.
    std::vector<Point> particles(100);
    for (std::size_t i = 0; i != particles.size(); ++i) {
      for (std::size_t j = 0; j != _Dimension; ++j) {
        particles[i][j] = random();
      }
    }

    // Order the particles.
    mortonOrder.setParticles(particles.begin(), particles.end());
  }

  // Check the result.
  assert(std::is_sorted(mortonOrder._cellCodes.begin(),
                        mortonOrder._cellCodes.end()));
  // Loop over the cells.
  for (std::size_t cell = 0; cell != mortonOrder.cellsSize(); ++cell) {
    const Code code = mortonOrder._cellCodes[cell];
    // Loop over the particles in the cell.
    for (std::size_t i = mortonOrder.cellBegin(cell);
         i != mortonOrder.cellEnd(cell); ++i) {
      // Check the code.
      assert(mortonOrder.morton.code(mortonOrder.particles[i]) == code);
    }
  }
}

int
main()
{
  {
    float length = 0.125;
    for (std::size_t i = 0; i != 5; ++i, length *= 0.5) {
      testPadding<1, false>(length);
      testPadding<2, false>(length);
      testPadding<3, false>(length);
      testPadding<1, true>(length);
      testPadding<2, true>(length);
      testPadding<3, true>(length);
    }
    length = 1;
    for (std::size_t i = 0; i != 5; ++i, length *= 0.5) {
      test<1, false>(length, float(0));
      test<2, false>(length, float(0));
      test<3, false>(length, float(0));
      test<1, true>(float(0.25) * length, float(0));
      test<2, true>(float(0.25) * length, float(0));
      test<3, true>(float(0.25) * length, float(0));

      test<1, false>(float(0.5) * length, float(0.5) * length);
      test<2, false>(float(0.5) * length, float(0.5) * length);
      test<3, false>(float(0.5) * length, float(0.5) * length);
      test<1, true>(float(0.125) * length, float(0.125) * length);
      test<2, true>(float(0.125) * length, float(0.125) * length);
      test<3, true>(float(0.125) * length, float(0.125) * length);
    }
  }
  {
    double length = 0.125;
    for (std::size_t i = 0; i != 5; ++i, length *= 0.5) {
      testPadding<1, false>(length);
      testPadding<2, false>(length);
      testPadding<3, false>(length);
      testPadding<1, true>(length);
      testPadding<2, true>(length);
      testPadding<3, true>(length);
    }
    length = 1;
    for (std::size_t i = 0; i != 5; ++i, length *= 0.5) {
      test<1, false>(length, double(0));
      test<2, false>(length, double(0));
      test<3, false>(length, double(0));
      test<1, true>(0.25 * length, double(0));
      test<2, true>(0.25 * length, double(0));
      test<3, true>(0.25 * length, double(0));

      test<1, false>(0.5 * length, 0.5 * length);
      test<2, false>(0.5 * length, 0.5 * length);
      test<3, false>(0.5 * length, 0.5 * length);
      test<1, true>(0.125 * length, 0.125 * length);
      test<2, true>(0.125 * length, 0.125 * length);
      test<3, true>(0.125 * length, 0.125 * length);
    }
  }

  return 0;
}
