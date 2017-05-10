// -*- C++ -*-

#include "stlib/particle/order.h"
#include "stlib/particle/traits.h"

#if defined(NEIGHBORS_VERLET)
#include "stlib/particle/verlet.h"
#elif defined(NEIGHBORS_ADJACENT)
#include "stlib/particle/adjacent.h"
#elif defined(NEIGHBORS_UNION)
#include "stlib/particle/unionMask.h"
#else
#error Bad value for neighbors method.
#endif

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/numerical/partition.h"

#include <fstream>

#include <string>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
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

#if 0
void
printSizes(const std::size_t size, const char* message)
{
  std::vector<std::size_t> sizes(MPI::COMM_WORLD.Get_size());
  MPI::COMM_WORLD.Gather(&size, 1, MPI::LONG, &sizes[0], 1, MPI::LONG, 0);
  if (MPI::COMM_WORLD.Get_rank() == 0) {
    std::cout << message << ": " << sizes[0];
    for (std::size_t i = 1; i != sizes.size(); ++i) {
      std::cout << ", " << sizes[i];
    }
    std::cout << '\n';
  }
}
#endif

template<typename _Float, std::size_t _Dimension, bool _Periodic>
void
run(const std::size_t manifoldDimension, const _Float interactionDistance,
    const _Float padding, const _Float velocity, const std::size_t numParticles,
    const std::size_t numSteps)
{
  typedef typename particle::TemplatedTypes<_Float, _Dimension>::Point Point;
  typedef Particle<_Float, _Dimension> Particle;
  typedef particle::Traits<Particle, GetPosition<_Float, _Dimension>,
          SetPosition<_Float, _Dimension>, _Periodic,
          _Dimension, _Float> Traits;
  typedef particle::MortonOrder<Traits> MortonOrder;
#if defined(NEIGHBORS_VERLET)
  typedef particle::VerletLists<MortonOrder> Neighbors;
#elif defined(NEIGHBORS_ADJACENT)
  typedef particle::AdjacentMask<MortonOrder> Neighbors;
#elif defined(NEIGHBORS_UNION)
  typedef particle::UnionMask<MortonOrder> Neighbors;
#else
#error Bad value for neighbors method.
#endif
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  const geom::BBox<_Float, _Dimension> Domain =
  {ext::filled_array<Point>(0), ext::filled_array<Point>(1)};
  MortonOrder mortonOrder(Domain, interactionDistance, padding);
  Neighbors neighbors(mortonOrder);

#ifdef _OPENMP
  std::size_t NumThreads;
  #pragma omp parallel
  if (omp_get_thread_num() == 0) {
    NumThreads = omp_get_num_threads();
  }
#else
  std::size_t NumThreads = 1;
#endif

  // The random number generators. Seed with the thread ID's.
  std::vector<ContinuousUniformGenerator::DiscreteUniformGenerator>
  generators(NumThreads);
  for (std::size_t i = 0; i != generators.size(); ++i) {
    generators[i].seed(i);
  }

  // Make a random vector of particles with random positions.
  std::vector<Particle> particles(numParticles);
  {
    ContinuousUniformGenerator random(&generators[0]);
    for (std::size_t i = 0; i != particles.size(); ++i) {
      // Pick a random point in the domain.
      for (std::size_t j = 0; j != manifoldDimension; ++j) {
        particles[i].position[j] = Domain.lower[j] +
                                   random() * (Domain.upper[j] - Domain.lower[j]);
      }
      for (std::size_t j = manifoldDimension; j != _Dimension; ++j) {
        particles[i].position[j] = 0;
      }
      particles[i].identifier = i;
      particles[i].interactionCount = 0;
    }
  }

  // Calculate the morton order.
  mortonOrder.setParticles(particles.begin(), particles.end());

  ads::Timer timer;
  double timeMoveParticles = 0;
  double timeCountNeighbors = 0;

  for (std::size_t n = 0; n != numSteps; ++n) {
    mortonOrder.repair();
    // Find the neighbors.
    neighbors.findLocalNeighbors();

    // Count the number of neighbors.
    timer.tic();
    #pragma omp parallel for
    for (std::ptrdiff_t i = mortonOrder.localParticlesBegin();
         i < std::ptrdiff_t(mortonOrder.localParticlesEnd()); ++i) {
      // Count the number of neighbors.
#if defined(NEIGHBORS_VERLET)
      mortonOrder.particles[i].interactionCount +=
        neighbors.neighbors.size(i);
#else
      mortonOrder.particles[i].interactionCount +=
        neighbors.numNeighbors(i);
#endif
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
      ContinuousUniformGenerator random(&generators[threadId]);

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
  mortonOrder.printPerformanceInfo(std::cout);
  neighbors.printPerformanceInfo(std::cout);

  // Check the sum of the identifiers.
  std::size_t sumIds = 0;
  for (std::size_t i = mortonOrder.localParticlesBegin();
       i != mortonOrder.localParticlesEnd();
       ++i) {
    sumIds += mortonOrder.particles[i].identifier;
  }
  assert(sumIds == particles.size() * (particles.size() - 1) / 2);
  // Sum the interactions.
  std::size_t localInteractions = 0;
  for (std::size_t i = mortonOrder.localParticlesBegin();
       i != mortonOrder.localParticlesEnd();
       ++i) {
    localInteractions += mortonOrder.particles[i].interactionCount;
  }
  // Print info about the non-orthtree parts of the simulation.
  std::cout << "\nSimulation costs:\n"
            << "MoveParticles,CountNeighbors\n"
            << timeMoveParticles << ',' << timeCountNeighbors << '\n'
            << "\nInteractions per particle per step = "
            << localInteractions / double(particles.size() * numSteps) << '\n';
}


// The program name.
std::string programName;

// Exit with an error message.
void
helpMessage()
{
  std::cout
      << "Usage:\n"
      << programName
      << " [-t=T] [-h] [-d=D] [-p=P] [-n=N] [-i=I] [-f=F] [-v=V] [-s=S] [-m=M]\n"
      << "-t: The number of threads.\n"
      << "-h: Print this help message and exit.\n"
      << "-d: The dimension of the manifold for the initial distribution of\n"
      << "    the points. The default is 3.\n"
      << "-p: The number of particles. The default is 1000.\n"
      << "-n: The average number of neighbors per particle. The default is 30.\n"
      << "    This parameter is used to set the interaction distance.\n"
      << "-i: The interaction distance.\n"
      << "-f: The padding is this fraction of the interaction distance.\n"
      << "-v: The velocity as a fraction of the interaction distance.\n"
      << "    The default is 0.01.\n"
      << "-s: The number of steps.\n"
      << "-m: The mode may be \"plain\" or \"periodic.\"\n";
  exit(0);
}

int
main(int argc, char* argv[])
{
  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();

#ifdef _OPENMP
  std::size_t numThreads = 0;
  if (parser.getOption('t', &numThreads)) {
    assert(numThreads > 0);
    omp_set_num_threads(numThreads);
  }
#endif

#ifdef _OPENMP
  #pragma omp parallel
  if (omp_get_thread_num() == 0) {
    std::cout << "OpenMP num threads = " << omp_get_num_threads()
              << "\n\n";
  }
#else
  std::cout << "OpenMP is not available.\n\n";
#endif

  if (parser.getOption('h')) {
    helpMessage();
  }

  std::size_t manifoldDimension = 3;
  parser.getOption('d', &manifoldDimension);
  assert(manifoldDimension <= 3);

  std::size_t numParticles = 1000;
  parser.getOption('p', &numParticles);
  assert(numParticles > 0);

  // First set the interaction distance using the average number of neighbors.
  float numNeighbors = 30;
  parser.getOption('n', &numNeighbors);
  assert(numNeighbors > 0);
  // numNeighbors = density (4/3) pi r^3
  // r = ((3/4) numNeighbors / (pi density))^(1/3)
  float interactionDistance = std::pow(0.75 * numNeighbors /
                                       (3.14159 * numParticles),
                                       1. / 3);
  // This can be overriden by directly setting the interaction distance.
  parser.getOption('i', &interactionDistance);
  assert(interactionDistance > 0);

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
    run<float, 3, false>(manifoldDimension, interactionDistance, padding,
                         velocity, numParticles, numSteps);
  }
  else if (mode == "periodic") {
    run<float, 3, true>(manifoldDimension, interactionDistance, padding,
                        velocity, numParticles, numSteps);
  }
  else {
    assert(false);
  }

  return 0;
}
