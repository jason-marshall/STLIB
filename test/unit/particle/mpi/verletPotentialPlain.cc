// -*- C++ -*-

#include "stlib/particle/orderMpi.h"
#include "stlib/particle/traits.h"
#include "stlib/particle/verletPotential.h"
#include "stlib/ads/functor/Identity.h"
#include "stlib/container/SimpleMultiIndexRangeIterator.h"

using namespace stlib;

void
printSizes(const std::size_t size)
{
  std::vector<std::size_t> const sizes = mpi::gather(size);
  if (mpi::commRank() == 0) {
    std::cout << "sizes = " << sizes[0];
    for (std::size_t i = 1; i != sizes.size(); ++i) {
      std::cout << ", " << sizes[i];
    }
    std::cout << '\n';
  }
}

template<typename _Float>
void
test()
{
  const std::size_t Dimension = 3;
  typedef std::array<_Float, Dimension> Point;
  typedef Point Particle;
  typedef container::SimpleMultiIndexRange<Dimension> Range;
  typedef typename Range::IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<Dimension> IndexIterator;
  typedef particle::PlainTraits<Particle, ads::Identity<Particle>,
          Dimension, _Float> Traits;
  typedef particle::MortonOrderMpi<Traits> MortonOrder;

  const std::size_t commRank = mpi::commRank();

  const std::size_t Extent = 3;
  const geom::BBox<_Float, Dimension> Domain =
  {ext::filled_array<Point>(0), ext::filled_array<Point>(Extent - 1)};

  MortonOrder mortonOrder(MPI_COMM_WORLD, Domain, 1.1, 1.1, 0.);
  particle::VerletListsPotential<MortonOrder> verlet(mortonOrder);

  // The particles are at lattice points. Define them on process 0.
  std::vector<Particle> particles;
  if (commRank == 0) {
    const Range range = {ext::filled_array<IndexList>(Extent),
                         ext::filled_array<IndexList>(0)
                        };
    const IndexIterator end = IndexIterator::end(range);
    for (IndexIterator i = IndexIterator::begin(range); i != end; ++i) {
      particles.push_back(ext::convert_array<_Float>(*i));
    }
  }

  // Calculate the morton order.
  mortonOrder.setParticles(particles.begin(), particles.end());
  mortonOrder.exchangeParticles();
  mortonOrder.checkValidity();
  if (commRank == 0) {
    std::cout << "Numbers of particles:\n";
  }
  printSizes(mortonOrder.particles.size());

  // Check the total number of potential neighbors.
  verlet.findPotentialNeighbors(mortonOrder.localCellsBegin(),
                                mortonOrder.localCellsEnd());
  std::size_t count = verlet.numPotentialNeighbors();
  if (commRank == 0) {
    std::cout << "Numbers of neighbors:\n";
  }
  printSizes(count);
  const std::size_t globalCount = mpi::reduce(count, MPI_SUM);
  if (commRank == 0) {
    const std::size_t n = Extent;
    const std::size_t target = 6 * (n - 2) * (n - 2) * (n - 2) + // interior
                               5 * (n - 2) * (n - 2) * 6 + // faces
                               4 * (n - 2) * 12 + // edges
                               3 * 8; // corners
    assert(globalCount == target);
  }
}


int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  test<float>();

  MPI_Finalize();
  return 0;
}
