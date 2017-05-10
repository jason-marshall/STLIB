// -*- C++ -*-

#include "stlib/particle/particle.h"
#include "stlib/particle/orderMpi.h"
#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/numerical/partition.h"

#include <fstream>

#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

template<typename _Float, std::size_t _Dimension>
struct Particle {
  std::array<_Float, _Dimension> position;
  std::size_t identifier;
  std::size_t interactionCount;
};

template<typename _Float, std::size_t _Dimension>
struct GetPosition :
    public std::unary_function<Particle<_Float, _Dimension>,
    std::array<_Float, _Dimension> > {
  typedef std::unary_function<Particle<_Float, _Dimension>,
          std::array<_Float, _Dimension> > Base;
  const typename Base::result_type&
  operator()(const typename Base::argument_type& x) const
  {
    return x.position;
  }
};

template<typename _Float, std::size_t _Dimension>
struct SetPosition :
    public std::binary_function<Particle<_Float, _Dimension>*,
    std::array<_Float, _Dimension>, void> {
  typedef std::binary_function<Particle<_Float, _Dimension>*,
          std::array<_Float, _Dimension>, void> Base;
  typename Base::result_type
  operator()(typename Base::first_argument_type particle,
             const typename Base::second_argument_type& point) const
  {
    particle->position = point;
  }
};

void
printSizes(const std::size_t size, const char* message)
{
  std::vector<std::size_t> const sizes = mpi::gather(size);
  if (mpi::commRank() == 0) {
    std::cout << message << ": " << sizes[0];
    for (std::size_t i = 1; i != sizes.size(); ++i) {
      std::cout << ", " << sizes[i];
    }
    std::cout << '\n';
  }
}

template<typename _Float, std::size_t _Dimension, bool _Periodic>
void
run(const std::size_t manifoldDimension, const _Float interactionDistance,
    const _Float shadowWidth, const _Float padding, const _Float velocity,
    const std::size_t numParticles, const std::size_t numSteps,
    const std::string& cellData)
{
  typedef particle::IntegerTypes::Code Code;
  typedef typename particle::TemplatedTypes<_Float, _Dimension>::Point Point;
  typedef Particle<_Float, _Dimension> Particle;
  typedef particle::Traits<Particle, GetPosition<_Float, _Dimension>,
          SetPosition<_Float, _Dimension>, _Periodic,
          _Dimension, _Float> Traits;
  typedef particle::MortonOrderMpi<Traits> MortonOrder;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  const std::size_t commRank = mpi::commRank();
  const std::size_t commSize = mpi::commSize();
  if (commRank == 0) {
    std::cout << "_Dimension = " << _Dimension
              << ", numParticles = " << numParticles << '\n';
  }

  const geom::BBox<_Float, _Dimension> Domain =
  {ext::filled_array<Point>(0), ext::filled_array<Point>(1)};
  MortonOrder mortonOrder(MPI_COMM_WORLD, Domain, interactionDistance,
                          shadowWidth, padding);
  particle::VerletLists<MortonOrder> verlet(mortonOrder);

  // Make a Morton object for a partitioning of the domain that will be used
  // for particle generation.
  particle::Morton<_Float, _Dimension, true> initial(Domain, 1);
  {
    // Choose the number of levels so that the number of cells is exactly
    // divisible by the number of processes or is significantly larger
    // than it.
    std::size_t levels = 0;
    while (std::size_t(1) << (levels * _Dimension) < commSize) {
      ++levels;
    }
    if ((std::size_t(1) << (levels * _Dimension)) % commSize != 0) {
      levels += 2;
    }
    initial.setLevels(levels);
  }

  // Our partition of the codes.
  Code codesLower, codesUpper;
  numerical::getPartitionRange(initial.maxCode() + 1, Code(commSize),
                               Code(commRank),
                               &codesLower, &codesUpper);

  if (commRank == 0) {
    std::cout << "\nInitial extents = " << initial.cellExtents() << '\n'
              << "Initial lengths = " << initial.lengths() << '\n'
              << "Code range = " << codesLower << ' ' << codesUpper << '\n'
              << '\n';
  }

#ifdef _OPENMP
  std::size_t NumThreads;
  #pragma omp parallel
  if (omp_get_thread_num() == 0) {
    NumThreads = omp_get_num_threads();
  }
#else
  std::size_t NumThreads = 1;
#endif

  // The random number generators. Seed with the ranks.
  std::vector<ContinuousUniformGenerator::DiscreteUniformGenerator>
  generators(NumThreads);
  for (std::size_t i = 0; i != generators.size(); ++i) {
    generators[i].seed(commRank * NumThreads + i);
  }
  //ContinuousUniformGenerator random(&generators[0]);

  // Make a random vector of particles with random positions.
  std::vector<Particle> particles(numParticles);
  #pragma omp parallel
  {
#ifdef _OPENMP
    const std::size_t threadId = omp_get_thread_num();
#else
    const std::size_t threadId = 0;
#endif
    ContinuousUniformGenerator::DiscreteUniformGenerator& generator =
      generators[threadId];
    ContinuousUniformGenerator random(&generator);
    std::size_t begin, end;
    numerical::getPartitionRange(particles.size(), &begin, &end);
    for (std::size_t i = begin; i != end; ++i) {
      // Pick a cell.
      const Code code = codesLower +
                        generator() % (codesUpper - codesLower);
      const Point lowerCorner = initial.cellLengths() *
                                ext::convert_array<_Float>(initial.coordinates(code));
      // Pick a random point in the cell.
      for (std::size_t j = 0; j != manifoldDimension; ++j) {
        particles[i].position[j] = lowerCorner[j] +
                                   random() * initial.cellLengths()[j];
      }
      for (std::size_t j = manifoldDimension; j != _Dimension; ++j) {
        particles[i].position[j] = 0;
      }
      particles[i].identifier = commRank * particles.size() + i;
      particles[i].interactionCount = 0;
    }
  }

  // Calculate the morton order.
  mortonOrder.setParticles(particles.begin(), particles.end());
  mortonOrder.exchangeParticles();
  //printSizes(mortonOrder.size(), "Number of particles");

  ads::Timer timer;
  double timeMoveParticles = 0;
  double timeCountNeighbors = 0;

  for (std::size_t n = 0; n != numSteps; ++n) {
    mortonOrder.repair();
    mortonOrder.exchangeParticles();
    verlet.findLocalNeighbors();

    timer.tic();
    #pragma omp parallel for
    for (std::ptrdiff_t i = mortonOrder.localParticlesBegin();
         i < std::ptrdiff_t(mortonOrder.localParticlesEnd()); ++i) {
      // Count the number of neighbors.
      mortonOrder.particles[i].interactionCount += verlet.neighbors.size(i);
    }
    timeCountNeighbors += timer.toc();

    timer.tic();
    #pragma omp parallel
    {
#ifdef _OPENMP
      const std::size_t threadId = omp_get_thread_num();
#else
      const std::size_t threadId = 0;
#endif
      ContinuousUniformGenerator::DiscreteUniformGenerator& generator =
        generators[threadId];
      ContinuousUniformGenerator random(&generator);

      Point v;
      std::size_t begin, end;
      numerical::getPartitionRange(mortonOrder.localParticlesEnd() -
                                   mortonOrder.localParticlesBegin(),
                                   &begin, &end);
      begin += mortonOrder.localParticlesBegin();
      end += mortonOrder.localParticlesBegin();
      for (std::size_t i = begin; i != end; ++i) {
        // Determine a displacement vector.
        for (std::size_t j = 0; j != _Dimension; ++j) {
          v[j] = 2 * random() - 1;
        }
        stlib::ext::normalize(&v);
        v *= velocity;
        // Move the particle.
        mortonOrder.particles[i].position += v;
        if (! _Periodic) {
          // Keep the particle inside the unit box.
          for (std::size_t j = 0; j != _Dimension; ++j) {
            _Float& x = mortonOrder.particles[i].position[j];
            if (x < 0) {
              x = 0;
            }
            if (x > 1) {
              x = 1;
            }
          }
        }
      }
    }
    timeMoveParticles += timer.toc();
  }
  //printSizes(mortonOrder.size(), "Number of particles");
  mortonOrder.printPerformanceInfo(std::cout);
  if (commRank == 0) {
    verlet.printPerformanceInfo(std::cout);
  }

  // Check the sum of the identifiers.
  std::size_t sumIds = 0;
  for (std::size_t i = mortonOrder.localParticlesBegin();
       i != mortonOrder.localParticlesEnd(); ++i) {
    sumIds += mortonOrder.particles[i].identifier;
  }
  std::size_t const globalSum = mpi::reduce(sumIds, MPI_SUM);
  if (commRank == 0) {
    const std::size_t n = commSize * particles.size();
    if (globalSum != n * (n - 1) / 2) {
      throw std::runtime_error("Error: The sum of the identifiers is "
                               "incorrect.");
    }
  }
  // Sum the interactions.
  std::size_t localInteractions = 0;
  for (std::size_t i = mortonOrder.localParticlesBegin();
       i != mortonOrder.localParticlesEnd(); ++i) {
    localInteractions += mortonOrder.particles[i].interactionCount;
  }
  std::size_t const globalInteractions =
    mpi::reduce(localInteractions, MPI_SUM);
  // Print info about the serial part of the simulation.
  if (commRank == 0) {
    std::cout << "\nSimulation costs:\n"
              << ",MoveParticles,CountNeighbors\n"
              << timeMoveParticles << ',' << timeCountNeighbors << '\n'
              << "\nInteractions per particle per step = "
              << globalInteractions /
              double(particles.size() * commSize * numSteps) << '\n';
  }
  // Write the cell data.
  if (! cellData.empty()) {
    std::ofstream f(cellData.c_str());
    mortonOrder.printCellDataVtk(f);
    f.close();
  }
}


// The program name.
std::string programName;

// Exit with an error message.
void
helpMessage()
{
  if (mpi::commRank() == 0) {
    std::cout
        << "Usage:\n"
        << programName
        << " [cellData] [-t=T] [-h] [-d D] [-p P] [-n N] [-i I] [-f F] [-v V] [-s S] [-m M]\n"
        << "-t: The number of threads.\n"
        << "cellData is a VTK file containing cell data.\n"
        << "-h: Print this help message and exit.\n"
        << "-d: The dimension of the manifold for the initial distribution of\n"
        << "    the points. The default is 3.\n"
        << "-p: The number of particles per process. The default is 1000.\n"
        << "-n: The average number of neighbors per particle. The default is 30.\n"
        << "    This parameter is used to set the interaction distance.\n"
        << "-i: The interaction distance.\n"
        << "-f: The padding is this fraction of the interaction distance.\n"
        << "-v: The velocity as a fraction of the interaction distance.\n"
        << "    The default is 0.01.\n"
        << "-s: The number of steps.\n"
        << "-m: The mode may be \"plain\" or \"periodic.\"\n";
  }
  MPI_Finalize();
  exit(0);
}

int
main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);

  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();

#ifdef _OPENMP
  std::size_t numThreads = 0;
  if (parser.getOption('t', &numThreads)) {
    assert(numThreads > 0);
    omp_set_num_threads(numThreads);
  }
#endif

  if (mpi::commRank() == 0) {
    std::cout << "Num MPI processes = " << mpi::commSize() << '\n';
#ifdef _OPENMP
    #pragma omp parallel
    if (omp_get_thread_num() == 0) {
      std::cout << "OpenMP num threads = " << omp_get_num_threads()
                << "\n\n";
    }
#else
    std::cout << "OpenMP is not available.\n\n";
#endif
  }

  std::string cellData;
  if (! parser.areArgumentsEmpty()) {
    cellData = parser.getArgument();
  }

  if (parser.getOption('h')) {
    helpMessage();
  }

  std::size_t manifoldDimension = 3;
  parser.getOption('d', &manifoldDimension);
  assert(manifoldDimension <= 3);

  std::size_t numParticlesPerProcess = 1000;
  parser.getOption('p', &numParticlesPerProcess);
  assert(numParticlesPerProcess > 0);

  // First set the interaction distance using the average number of neighbors.
  float numNeighbors = 30;
  parser.getOption('n', &numNeighbors);
  assert(numNeighbors > 0);
  // numNeighbors = density (4/3) pi r^3
  // r = ((3/4) numNeighbors / (pi density))^(1/3)
  float interactionDistance = std::pow(0.75 * numNeighbors /
                                       (3.14159 * mpi::commSize() *
                                        numParticlesPerProcess), 1. / 3);
  // This can be overriden by directly setting the interaction distance.
  parser.getOption('i', &interactionDistance);
  assert(interactionDistance > 0);

  float shadowWidth = interactionDistance;
  parser.getOption('w', &shadowWidth);
  assert(shadowWidth >= interactionDistance);

  float padding = std::numeric_limits<float>::quiet_NaN();
  float fraction = 0;
  if (parser.getOption('f', &fraction)) {
    padding = fraction * interactionDistance;
  }
  assert(padding != padding || padding >= 0);

  float velocity = 0.01 * interactionDistance;
  if (parser.getOption('v', &fraction)) {
    velocity = fraction * interactionDistance;
  }
  assert(velocity >= 0);

  std::size_t numSteps = 1000;
  parser.getOption('s', &numSteps);
  assert(numSteps > 0);

  std::string mode;
  parser.getOption('m', &mode);
  if (mode.empty() || mode == "plain") {
    run<float, 3, false>(manifoldDimension, interactionDistance, shadowWidth,
                         padding, velocity, numParticlesPerProcess, numSteps,
                         cellData);
  }
  else if (mode == "periodic") {
    run<float, 3, true>(manifoldDimension, interactionDistance, shadowWidth,
                        padding, velocity, numParticlesPerProcess, numSteps,
                        cellData);
  }
  else {
    assert(false);
  }

  MPI_Finalize();
  return 0;
}
