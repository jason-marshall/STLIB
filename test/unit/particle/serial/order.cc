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

  // Cached positions.
  {
    std::vector<Point> cachedPositions;
    mortonOrder.getPositions(&cachedPositions);
    assert(cachedPositions.size() == mortonOrder.particles.size());
    for (std::size_t i = 0; i != mortonOrder.particles.size(); ++i) {
      assert(cachedPositions[i] == mortonOrder.position(i));
    }
  }

  // Cells.
  assert(mortonOrder.cellsSize() > 0);
  for (std::size_t i = 0; i != mortonOrder.cellsSize(); ++i) {
    // The cell codes must be increasing. (Note that at the end of the
    // loop we use the guard element.)
    assert(mortonOrder._cellCodes[i] < mortonOrder._cellCodes[i + 1]);
    assert(mortonOrder.cellBegin(i) < mortonOrder.cellEnd(i));
    for (std::size_t j = mortonOrder.cellBegin(i);
         j != mortonOrder.cellEnd(i); ++j) {
      assert(mortonOrder.morton.code(mortonOrder.particles[j]) ==
             mortonOrder._cellCodes[i]);
    }
  }
}


void
testAdjacentCells2Plain()
{
  const std::size_t Dimension = 2;
  const bool Periodic = false;
  typedef float Float;
  typedef std::array<Float, Dimension> Particle;
  typedef particle::Traits<Particle, ads::Identity<Particle>,
          SetPosition<Float, Dimension>, Periodic,
          Dimension, Float> Traits;
  typedef MortonOrder<Traits> MortonOrder;
  typedef MortonOrder::Point Point;
  typedef particle::Neighbor<Periodic> Neighbor;
  typedef particle::NeighborCell<Periodic> NeighborCell;

  // Make the data structure for ordering the particles.
  const geom::BBox<Float, Dimension> Domain = {ext::filled_array<Point>(0),
                                               ext::filled_array<Point>(2)
                                              };
  const Float interactionDistance = 1;
  const Float padding = 0;
  MortonOrder mortonOrder(Domain, interactionDistance, padding);

  {
    // Make a vector of particles.
    std::vector<Point> particles;
    particles.push_back(Point{
      {
        0.5, 0.5
      }
    });
    particles.push_back(Point{
      {
        1.5, 0.5
      }
    });
    particles.push_back(Point{
      {
        0.5, 1.5
      }
    });
    particles.push_back(Point{
      {
        1.5, 1.5
      }
    });
    // Order the particles.
    mortonOrder.setParticles(particles.begin(), particles.end());
  }
  assert(mortonOrder.cellsSize() == 4);

  //
  // adjacentCells
  //
  assert(mortonOrder.adjacentCells.numArrays() == 4);
  for (std::size_t i = 0; i != mortonOrder.adjacentCells.numArrays(); ++i) {
    assert(mortonOrder.adjacentCells.size(i) == 4);
    assert(mortonOrder.numAdjacentNeighbors[i] == 4);
    assert(mortonOrder.countAdjacentParticles(i) == 4);
  }

  for (std::size_t i = 0; i != mortonOrder.adjacentCells.numArrays(); ++i) {
    for (std::size_t j = 0; j != mortonOrder.adjacentCells.size(i); ++j) {
      assert(mortonOrder.adjacentCells(i, j) == NeighborCell{
        j
      });
    }
  }

  //
  // positionsInAdjacent(cell, positions)
  //
  {
    std::vector<Point> positions;
    // Cell 0.
    mortonOrder.positionsInAdjacent(0, &positions);
    assert(positions.size() == mortonOrder.particles.size());
    assert(positions[0] == (Point{
      {
        0.5, 0.5
      }
    }));
    assert(positions[1] == (Point{
      {
        1.5, 0.5
      }
    }));
    assert(positions[2] == (Point{
      {
        0.5, 1.5
      }
    }));
    assert(positions[3] == (Point{
      {
        1.5, 1.5
      }
    }));
    // Cell 1.
    mortonOrder.positionsInAdjacent(1, &positions);
    assert(positions.size() == mortonOrder.particles.size());
    // Cell 2.
    mortonOrder.positionsInAdjacent(2, &positions);
    assert(positions.size() == mortonOrder.particles.size());
    // Cell 3.
    mortonOrder.positionsInAdjacent(3, &positions);
    assert(positions.size() == mortonOrder.particles.size());
  }
  //
  // positionsInAdjacent(cachedPositions, cell, positions)
  //
  {
    std::vector<Point> cachedPositions;
    mortonOrder.getPositions(&cachedPositions);
    std::vector<Point> positions;
    // Cell 0.
    mortonOrder.positionsInAdjacent(cachedPositions, 0, &positions);
    assert(positions.size() == mortonOrder.particles.size());
    assert(positions[0] == (Point{
      {
        0.5, 0.5
      }
    }));
    assert(positions[1] == (Point{
      {
        1.5, 0.5
      }
    }));
    assert(positions[2] == (Point{
      {
        0.5, 1.5
      }
    }));
    assert(positions[3] == (Point{
      {
        1.5, 1.5
      }
    }));
    // Cell 1.
    mortonOrder.positionsInAdjacent(cachedPositions, 1, &positions);
    assert(positions.size() == mortonOrder.particles.size());
    // Cell 2.
    mortonOrder.positionsInAdjacent(cachedPositions, 2, &positions);
    assert(positions.size() == mortonOrder.particles.size());
    // Cell 3.
    mortonOrder.positionsInAdjacent(cachedPositions, 3, &positions);
    assert(positions.size() == mortonOrder.particles.size());
  }
  //
  // positionsInAdjacent(cell, indices, positions)
  //
  {
    std::vector<Neighbor> indices;
    std::vector<Point> positions;
    // Cell 0.
    mortonOrder.positionsInAdjacent(0, &indices, &positions);
    assert(positions[0] == (Point{
      {
        0.5, 0.5
      }
    }));
    assert(positions[1] == (Point{
      {
        1.5, 0.5
      }
    }));
    assert(positions[2] == (Point{
      {
        0.5, 1.5
      }
    }));
    assert(positions[3] == (Point{
      {
        1.5, 1.5
      }
    }));
    // All cells.
    for (std::size_t i = 0; i != mortonOrder.adjacentCells.numArrays(); ++i) {
      mortonOrder.positionsInAdjacent(i, &indices, &positions);
      assert(indices.size() == mortonOrder.particles.size());
      assert(positions.size() == mortonOrder.particles.size());
      for (std::size_t j = 0; j != mortonOrder.adjacentCells.size(i); ++j) {
        assert(indices[j].particle == j);
      }
    }
  }
}


void
testAdjacentCells2Periodic4x4()
{
  const std::size_t Dimension = 2;
  const bool Periodic = true;
  typedef float Float;
  typedef std::array<Float, Dimension> Particle;
  typedef particle::Traits<Particle, ads::Identity<Particle>,
          SetPosition<Float, Dimension>, Periodic,
          Dimension, Float> Traits;
  typedef MortonOrder<Traits> MortonOrder;
  typedef MortonOrder::Point Point;
  typedef particle::NeighborCell<Periodic> NeighborCell;
  typedef particle::Neighbor<Periodic> Neighbor;

  // Make the data structure for ordering the particles.
  const geom::BBox<Float, Dimension> Domain = {ext::filled_array<Point>(0),
                                               ext::filled_array<Point>(4)
                                              };
  const Float interactionDistance = 1;
  const Float padding = 0;
  MortonOrder mortonOrder(Domain, interactionDistance, padding);

  {
    // Make a vector of particles.
    std::vector<Point> particles;
    for (std::size_t j = 0; j != 4; ++j) {
      for (std::size_t i = 0; i != 4; ++i) {
        particles.push_back(Point{
          {
            Float(i + 0.5), Float(j + 0.5)
          }
        });
      }
    }
    // Order the particles.
    mortonOrder.setParticles(particles.begin(), particles.end());
  }
  assert(mortonOrder.cellsSize() == 16);
  assert(mortonOrder.adjacentCells.numArrays() == 16);
  for (std::size_t i = 0; i != mortonOrder.adjacentCells.numArrays(); ++i) {
    assert(mortonOrder.adjacentCells.size(i) == 9);
    assert(mortonOrder.numAdjacentNeighbors[i] == 9);
    assert(mortonOrder.countAdjacentParticles(i) == 9);
  }

  /*
    Cell ordering:
    10 11 14 15
    08 09 12 13
    02 03 06 07
    00 01 04 05

    Offsets:
    6 7 8
    3 4 5
    0 1 2
  */
  assert((mortonOrder.adjacentCells(0, 0) == NeighborCell{
    15, 0
  }));
  assert((mortonOrder.adjacentCells(0, 1) == NeighborCell{
    10, 1
  }));
  assert((mortonOrder.adjacentCells(0, 2) == NeighborCell{
    11, 1
  }));
  assert((mortonOrder.adjacentCells(0, 3) == NeighborCell{
    5, 3
  }));
  assert((mortonOrder.adjacentCells(0, 4) == NeighborCell{
    0, 4
  }));
  assert((mortonOrder.adjacentCells(0, 5) == NeighborCell{
    1, 4
  }));
  assert((mortonOrder.adjacentCells(0, 6) == NeighborCell{
    7, 3
  }));
  assert((mortonOrder.adjacentCells(0, 7) == NeighborCell{
    2, 4
  }));
  assert((mortonOrder.adjacentCells(0, 8) == NeighborCell{
    3, 4
  }));

  //
  // positionsInAdjacent()
  //
  std::vector<Neighbor> indices;
  std::vector<Point> positions;
  // Cell 0.
  mortonOrder.positionsInAdjacent(0, &indices, &positions);
  assert(indices.size() == 9);
  assert(positions.size() == indices.size());
  for (std::size_t i = 0; i != indices.size(); ++i) {
    // The particles coincide with the cells.
    assert(indices[i].particle == (mortonOrder.adjacentCells(0, i).cell));
  }
  assert(positions[0] == (Point{
    {
      -0.5, -0.5
    }
  }));
  assert(positions[1] == (Point{
    {
      0.5, -0.5
    }
  }));
  assert(positions[2] == (Point{
    {
      1.5, -0.5
    }
  }));
  assert(positions[3] == (Point{
    {
      -0.5, 0.5
    }
  }));
  assert(positions[4] == (Point{
    {
      0.5, 0.5
    }
  }));
  assert(positions[5] == (Point{
    {
      1.5, 0.5
    }
  }));
  assert(positions[6] == (Point{
    {
      -0.5, 1.5
    }
  }));
  assert(positions[7] == (Point{
    {
      0.5, 1.5
    }
  }));
  assert(positions[8] == (Point{
    {
      1.5, 1.5
    }
  }));
}


void
testAdjacentCells2Periodic3x3()
{
  const std::size_t Dimension = 2;
  const bool Periodic = true;
  typedef float Float;
  typedef std::array<Float, Dimension> Particle;
  typedef particle::Traits<Particle, ads::Identity<Particle>,
          SetPosition<Float, Dimension>, Periodic,
          Dimension, Float> Traits;
  typedef MortonOrder<Traits> MortonOrder;
  typedef MortonOrder::Point Point;
  typedef particle::NeighborCell<Periodic> NeighborCell;

  // Make the data structure for ordering the particles.
  const geom::BBox<Float, Dimension> Domain = {ext::filled_array<Point>(0),
                                               Point{{3, 3}}};
  const Float interactionDistance = 1;
  const Float padding = 0;
  MortonOrder mortonOrder(Domain, interactionDistance, padding);
  assert((mortonOrder.lengths() == Point{
    {
      3, 3
    }
  }));

  {
    // Make a vector of particles.
    std::vector<Point> particles;
    for (std::size_t j = 0; j != 3; ++j) {
      for (std::size_t i = 0; i != 3; ++i) {
        particles.push_back(Point{
          {
            Float(i + 0.5), Float(j + 0.5)
          }
        });
      }
    }
    // Order the particles.
    mortonOrder.setParticles(particles.begin(), particles.end());
  }
  assert(mortonOrder.cellsSize() == 9);
  assert(mortonOrder.adjacentCells.numArrays() == 9);
  for (std::size_t i = 0; i != mortonOrder.adjacentCells.numArrays(); ++i) {
    assert(mortonOrder.adjacentCells.size(i) == 9);
    assert(mortonOrder.numAdjacentNeighbors[i] == 9);
    assert(mortonOrder.countAdjacentParticles(i) == 9);
  }

  /*
    Cell ordering:
    XX XX XX XX
    06 07 08 XX
    02 03 05 XX
    00 01 04 XX

    Offsets:
    6 7 8
    3 4 5
    0 1 2
  */
  assert((mortonOrder.adjacentCells(0, 0) == NeighborCell{
    8, 0
  }));
  assert((mortonOrder.adjacentCells(0, 1) == NeighborCell{
    6, 1
  }));
  assert((mortonOrder.adjacentCells(0, 2) == NeighborCell{
    7, 1
  }));
  assert((mortonOrder.adjacentCells(0, 3) == NeighborCell{
    4, 3
  }));
  assert((mortonOrder.adjacentCells(0, 4) == NeighborCell{
    0, 4
  }));
  assert((mortonOrder.adjacentCells(0, 5) == NeighborCell{
    1, 4
  }));
  assert((mortonOrder.adjacentCells(0, 6) == NeighborCell{
    5, 3
  }));
  assert((mortonOrder.adjacentCells(0, 7) == NeighborCell{
    2, 4
  }));
  assert((mortonOrder.adjacentCells(0, 8) == NeighborCell{
    3, 4
  }));
}

template<typename _Float, std::size_t _Dimension>
void
testDefault()
{
  // Default constructor, plain domain.
  {
    typedef std::array<_Float, _Dimension> Particle;
    typedef particle::PlainTraits<Particle, ads::Identity<Particle> > Traits;
    typedef particle::MortonOrder<Traits> MortonOrder;
    MortonOrder mortonOrder;
  }
  // Default constructor, periodic domain.
  {
    typedef std::array<_Float, _Dimension> Particle;
    typedef particle::PeriodicTraits<Particle, ads::Identity<Particle>,
            SetPosition<_Float, _Dimension> > Traits;
    typedef particle::MortonOrder<Traits> MortonOrder;
    MortonOrder mortonOrder;
  }
}

int
main()
{
  testDefault<float, 1>();
  testDefault<float, 2>();
  testDefault<float, 3>();

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

  testAdjacentCells2Plain();
  testAdjacentCells2Periodic4x4();
  testAdjacentCells2Periodic3x3();

  return 0;
}
