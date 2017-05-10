// -*- C++ -*-

#include "stlib/particle/particle.h"

#include "stlib/particle/verlet.h"
#include "stlib/particle/adjacent.h"
#include "stlib/particle/unionMask.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer.h"
#include "stlib/container/SimpleMultiIndexExtentsIterator.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/numerical/partition.h"

#include <fstream>
#include <string>

#include <cstdio>

#ifdef __SSE__
#include <pmmintrin.h>
#endif

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

template<typename _Float, std::size_t _Dimension>
struct Mol {
  typedef std::array<_Float, _Dimension> Point;
  Point r, rv, ra;
};

template<typename _Float, std::size_t _Dimension>
struct GetPosition :
    public std::unary_function<Mol<_Float, _Dimension>,
    std::array<_Float, _Dimension> > {
  typedef std::unary_function<Mol<_Float, _Dimension>,
          std::array<_Float, _Dimension> > Base;
  const typename Base::result_type&
  operator()(const typename Base::argument_type& x) const
  {
    return x.r;
  }
};

template<typename _Float, std::size_t _Dimension>
struct SetPosition :
    public std::binary_function<Mol<_Float, _Dimension>*,
    std::array<_Float, _Dimension>, void> {
  typedef std::binary_function<Mol<_Float, _Dimension>*,
          std::array<_Float, _Dimension>, void> Base;
  typename Base::result_type
  operator()(typename Base::first_argument_type particle,
             const typename Base::second_argument_type& point) const
  {
    particle->r = point;
  }
};


// A property that is averaged.
template<typename _Float>
struct Prop {
  static const std::size_t StepAvg = 100;

  _Float val, sum, sum2;

  void
  zero()
  {
    sum = 0;
    sum2 = 0;
  }

  void
  accumulate()
  {
    sum += val;
    sum2 += val * val;
  }

  void
  average()
  {
    sum /= StepAvg;
    sum2 = std::sqrt(std::max(sum2 / StepAvg - sum * sum, _Float(0)));
  }
};

// Properties that describe the state of the system.
template<typename _Float, std::size_t _Dimension>
struct State {
  typedef std::array<_Float, _Dimension> Point;

  std::size_t stepCount;
  // The number of molecules.
  std::size_t nMol;
  // Potential energy.
  _Float uSum;
  _Float virSum;
  // Sum of velocities.
  Point vSum;
  // Sum of squared velocities.
  _Float vvSum;
  Prop<_Float> kinEnergy, totEnergy, pressure;

  // The time step.
  static
  _Float
  deltaT()
  {
    return 0.005;
  }

  static
  _Float
  density()
  {
    return 0.8;
  }

  static
  _Float
  temperature()
  {
    return 1;
  }

  static
  _Float
  interactionDistance()
  {
    return std::pow(2., 1. / 6);
  }

  State(const std::size_t nMol_ = 0) :
    stepCount(0),
    nMol(nMol_)
  {
  }

  template<typename _Order>
  void
  evalProps(const _Order& order)
  {
    vSum.fill(0);
    vvSum = 0;
    for (std::size_t i = order.localParticlesBegin();
         i != order.localParticlesEnd(); ++i) {
      vSum += order.particles[i].rv;
      vvSum += stlib::ext::dot(order.particles[i].rv, order.particles[i].rv);
    }
    assert(nMol != 0);
    kinEnergy.val = 0.5 * vvSum / nMol;
    totEnergy.val = kinEnergy.val + uSum / nMol;
    pressure.val = density() * (vvSum + virSum) / (nMol * _Dimension);
  }


  void
  zero()
  {
    totEnergy.zero();
    kinEnergy.zero();
    pressure.zero();
  }

  void
  accumulate()
  {
    totEnergy.accumulate();
    kinEnergy.accumulate();
    pressure.accumulate();
  }

  void
  average()
  {
    totEnergy.average();
    kinEnergy.average();
    pressure.average();
  }

  void
  printSummary(FILE* fp)
  {
    fprintf(fp, "%5d %8.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f\n",
            int(stepCount), stepCount * deltaT(), stlib::ext::sum(vSum) / nMol,
            totEnergy.sum, totEnergy.sum2,
            kinEnergy.sum, kinEnergy.sum2,
            pressure.sum, pressure.sum2);
    fflush(fp);
  }
};


template<typename _Order, typename Neighbors, typename _Float,
         std::size_t _Dimension>
inline
void
singleStep(_Order* order, const Neighbors& neighbors,
           State<_Float, _Dimension>* state)
{
  ++state->stepCount;
  leapfrogStep1(State<_Float, _Dimension>::deltaT(), order);
  computeForces(order, neighbors, state);
  leapfrogStep2(State<_Float, _Dimension>::deltaT(), order);
  state->evalProps(*order);
  state->accumulate();
  if (state->stepCount % Prop<_Float>::StepAvg == 0) {
    state->average();
    state->printSummary(stdout);
    state->zero();
  }
}

template<typename _Order, typename _Float, std::size_t _Dimension>
inline
void
setupJob(_Order* order,
         State<_Float, _Dimension>* state,
         const std::array<std::size_t, _Dimension>& latticeExtents)
{
  initCoords(order, latticeExtents);
  initVels(order);
  initAccels(order);
  state->nMol = order->particles.size();
  state->zero();
}

template<typename _Order, typename _Float, std::size_t _Dimension>
inline
void
computeForces(_Order* order, const particle::VerletLists<_Order>& neighbors,
              State<_Float, _Dimension>* state)
{
  typedef typename _Order::Point Point;

  Point dr;
  _Float fcVal, rr, rri, rri3;

  for (std::size_t i = 0; i != order->particles.size(); ++i) {
    order->particles[i].ra = ext::filled_array<Point>(0);
  }

  state->uSum = 0;
  state->virSum = 0;
  // For each particle.
  for (std::size_t i = order->localParticlesBegin();
       i != order->localParticlesEnd(); ++i) {
    // For each neighbor of the particle.
    for (std::size_t j = 0; j != neighbors.neighbors.size(i); ++j) {
      dr = order->particles[i].r - neighbors.neighborPosition(i, j);
      rr = stlib::ext::dot(dr, dr);
      rri = _Float(1) / rr;
      rri3 = rri * rri * rri;
      fcVal = _Float(48) * rri3 * (rri3 - _Float(0.5)) * rri;
      order->particles[i].ra += fcVal * dr;
      if (i < neighbors.neighbors(i, j).particle) {
        state->uSum += 4. * rri3 * (rri3 - 1.) + 1.;
        state->virSum += fcVal * rr;
      }
    }
  }
}


// Generic implementation.
template<typename _Order, typename _Float, std::size_t _Dimension>
inline
void
computeForces(_Order* order, const particle::AdjacentMask<_Order>& neighbors,
              State<_Float, _Dimension>* state)
{
  typedef typename _Order::Point Point;
  typedef particle::AdjacentMask<_Order> Neighbors;
  typedef typename Neighbors::Mask Mask;
  const std::size_t MaskDigits = Neighbors::MaskDigits;

  Point dr;
  _Float fcVal, rr, rri, rri3;

  for (std::size_t i = 0; i != order->particles.size(); ++i) {
    order->particles[i].ra = ext::filled_array<Point>(0);
  }

  // The indices and positions of the potential neighbors.
  std::vector<std::size_t> indices;
  std::vector<Point> positions;

  state->uSum = 0;
  state->virSum = 0;
  // For each cell.
  for (std::size_t cell = order->localCellsBegin();
       cell != order->localCellsEnd(); ++cell) {
    // Cache the potential neighbor indices and positions.
    neighbors.positionsInAdjacent(cell, &indices, &positions);
    // For each particle in the cell.
    for (std::size_t i = order->cellBegin(cell); i != order->cellEnd(cell);
         ++i) {
      // The neighbor index in the sequence of potential neighbors.
      // Incremented in the inner loop.
      std::size_t j = 0;
      // For each block of potential neighbors.
      for (std::size_t block = 0; block != neighbors.neighborMasks.size(i);
           ++block) {
        Mask mask = neighbors.neighborMasks(i, block);
        // For each potential neighbor in the block.
        for (std::size_t d = 0; d != MaskDigits; ++d, ++j, mask >>= 1) {
          // Skip potential neighbors that are not within the interaction
          // distance.
          if ((mask & Mask(1)) == 0) {
            continue;
          }
          dr = order->particles[i].r - positions[j];
          rr = dot(dr, dr);
          rri = _Float(1) / rr;
          rri3 = rri * rri * rri;
          fcVal = _Float(48) * rri3 * (rri3 - _Float(0.5)) * rri;
          order->particles[i].ra += fcVal * dr;
          if (i < indices[j]) {
            state->uSum += 4. * rri3 * (rri3 - 1.) + 1.;
            state->virSum += fcVal * rr;
          }
        }
      }
    }
  }
}


// If AVX or SSE is enabled.
#ifdef __SSE__
// CONTINUE: Move to simd package.
#if 0
std::ostream&
operator<<(std::ostream& out, __m128 x)
{
  float a[4] __attribute__((aligned(16)));
  _mm_store_ps(a, x);
  return out << a[0] << ' '
         << a[1] << ' '
         << a[2] << ' '
         << a[3];
}
#endif

// SIMD Implementation.
template<typename _Order>
inline
void
computeForces(_Order* order, const particle::AdjacentMask<_Order>& neighbors,
              State<float, 3>* state)
{
  typedef simd::Vector<float>::Type Vector;
  typedef typename _Order::Point Point;
  typedef particle::AdjacentMask<_Order> Neighbors;
  typedef typename Neighbors::Mask Mask;
  const std::size_t MaskDigits = Neighbors::MaskDigits;
  const std::size_t Dimension = _Order::Dimension;
  // The number of single-precision floats in the register.
  const std::size_t VectorSize = simd::Vector<float>::Size;
  const Mask SimdMask = (1 << VectorSize) - 1;
  // We require this for our use of memcpy().
  assert(sizeof(Point) == Dimension * sizeof(float));

  Vector neighborMask, countMask, drx, dry, drz, fcVal, rr, rri, rri3, uSum,
         virSum;

  for (std::size_t i = 0; i != order->particles.size(); ++i) {
    order->particles[i].ra = ext::filled_array<Point>(0);
  }

  // The indices and positions of the potential neighbors.
  std::vector<std::size_t> indices;
  std::vector<Point> positions;
  // The shuffled coordinates.
  std::vector<float, simd::allocator<float> > shuffled;

  state->uSum = 0;
  state->virSum = 0;
  // For each cell.
  for (std::size_t cell = order->localCellsBegin();
       cell != order->localCellsEnd(); ++cell) {
    // Cache the potential neighbor indices and positions.
    neighbors.positionsInAdjacent(cell, &indices, &positions);
    // Shuffle the position coordinates so that we can use SSE operations.
    shuffled.resize(Dimension * positions.size());
    memcpy(&shuffled[0], &positions[0], positions.size() * sizeof(Point));
    simd::aosToHybridSoa<Dimension>(&shuffled);
    // For each particle in the cell.
    for (std::size_t i = order->cellBegin(cell); i != order->cellEnd(cell);
         ++i) {
      const Point& p = order->particles[i].r;
      const Vector px = simd::set1(p[0]);
      const Vector py = simd::set1(p[1]);
      const Vector pz = simd::set1(p[2]);
      const float* block = &shuffled[0];
      // The neighbor index in the sequence of potential neighbors.
      // Incremented in the inner loop.
      std::size_t j = 0;
      // For each block of potential neighbors.
      for (std::size_t maskIndex = 0;
           maskIndex != neighbors.neighborMasks.size(i); ++maskIndex) {
        Mask mask = neighbors.neighborMasks(i, maskIndex);
        // For each SIMD block in the mask.
        for (std::size_t n = 0; n != MaskDigits / VectorSize; ++n,
             j += VectorSize, mask >>= VectorSize,
             block += Dimension * VectorSize) {
          // Skip blocks of potential neighbors that are not within the
          // interaction distance.
          if ((mask & SimdMask) == 0) {
            continue;
          }
#ifdef STLIB_AVX512F
#error Not implemented.
#elif defined(__AVX__)
          neighborMask = _mm256_set_ps
                         (mask & 128, mask & 64, mask & 32, mask & 16,
                          mask & 8, mask & 4, mask & 2, mask & 1);
#else
          neighborMask = _mm_set_ps(mask & 8, mask & 4, mask & 2, mask & 1);
#endif
          neighborMask = simd::notEqual(neighborMask, simd::setzero<float>());
          drx = simd::bitwiseAnd(neighborMask, px - simd::load(block));
          dry = simd::bitwiseAnd(neighborMask, py -
                                 simd::load(block + VectorSize));
          drz = simd::bitwiseAnd(neighborMask, pz -
                                 simd::load(block + 2 * VectorSize));
          rr = drx * drx + dry * dry + drz * drz;
          rr = simd::bitwiseAnd(neighborMask, rr);
          rri = simd::set1(float(1)) / rr;
          rri = simd::bitwiseAnd(neighborMask, rri);
          rri3 = rri * rri * rri;
          fcVal = simd::set1(float(48)) * rri3 *
                  (rri3 - simd::set1(float(0.5))) * rri;


#if defined(STLIB_AVX512F) || defined(__AVX__)
          order->particles[i].ra[0] += simd::sum(fcVal * drx);
          order->particles[i].ra[1] += simd::sum(fcVal * dry);
          order->particles[i].ra[2] += simd::sum(fcVal * drz);
#else
          ALIGN_SIMD float ab[VectorSize];
          // x0+x1+x2+x3, y0+y1+y2+y3, z0+z1, z2+z3
          simd::store(ab, _mm_hadd_ps(_mm_hadd_ps(fcVal * drx,
                                                  fcVal * dry),
                                      fcVal * drz));
          order->particles[i].ra[0] += ab[0];
          order->particles[i].ra[1] += ab[1];
          order->particles[i].ra[2] += ab[2] + ab[3];
#endif

#ifdef STLIB_AVX512F
#error Not implemented.
#elif defined(__AVX__)
          countMask = simd::notEqual(_mm256_set_ps(i < indices[j + 7],
                                     i < indices[j + 6],
                                     i < indices[j + 5],
                                     i < indices[j + 4],
                                     i < indices[j + 3],
                                     i < indices[j + 2],
                                     i < indices[j + 1],
                                     i < indices[j]),
                                     simd::setzero<float>());
#else
          countMask = simd::notEqual(_mm_set_ps(i < indices[j + 3],
                                                i < indices[j + 2],
                                                i < indices[j + 1],
                                                i < indices[j]),
                                     simd::setzero<float>());
#endif
          countMask = simd::bitwiseAnd(neighborMask, countMask);
          uSum = simd::bitwiseAnd(countMask,
                                  (simd::set1(float(4)) * rri3 *
                                   (rri3 - simd::set1(float(1))) +
                                   simd::set1(float(1))));
          virSum = simd::bitwiseAnd(countMask, fcVal * rr);
#ifdef STLIB_AVX512F
#error Not implemented.
#elif defined(__AVX__)
          state->uSum += simd::sum(uSum);
          state->virSum += simd::sum(virSum);
#else
          simd::store(ab, _mm_hadd_ps(_mm_hadd_ps(uSum, virSum),
                                      simd::setzero<float>()));
          state->uSum += ab[0];
          state->virSum += ab[1];
#endif
        }
      }
    }
  }
}
#endif


template<typename _Float, typename _Order>
inline
void
leapfrogStep1(const _Float DeltaT, _Order* order)
{
  for (std::size_t i = 0; i != order->particles.size(); ++i) {
    order->particles[i].rv += _Float(0.5) * DeltaT * order->particles[i].ra;
    order->particles[i].r += DeltaT * order->particles[i].rv;
  }
}


template<typename _Float, typename _Order>
inline
void
leapfrogStep2(const _Float DeltaT, _Order* order)
{
  for (std::size_t i = 0; i != order->particles.size(); ++i) {
    order->particles[i].rv += _Float(0.5) * DeltaT * order->particles[i].ra;
  }
}


template<typename _Order, std::size_t _Dimension>
inline
void
initCoords(_Order* order,
           const std::array<std::size_t, _Dimension>& latticeExtents)
{
  typedef typename _Order::Point Point;
  typedef typename _Order::Float Float;
  typedef container::SimpleMultiIndexExtentsIterator<_Dimension> IndexIterator;

  // Allocate a vector of particles.
  std::vector<typename _Order::Particle>
    particles(stlib::ext::product(latticeExtents));
  // Set the initial positions to be the centers of the voxels.
  const Point gap = order->lengths() /
                    ext::convert_array<Float>(latticeExtents);
  std::size_t n = 0;
  const IndexIterator end = IndexIterator::end(latticeExtents);
  for (IndexIterator i = IndexIterator::begin(latticeExtents); i != end;
       ++i, ++n) {
    particles[n].r = (ext::convert_array<Float>(*i) + Float(0.5)) * gap;
  }
  // Set the particles in the orthtree.
  order->setParticles(particles.begin(), particles.end());
}


template<typename _Order>
inline
void
initVels(_Order* order)
{
  typedef typename _Order::Point Point;
  typedef typename _Order::Float Float;
  typedef numerical::ContinuousUniformGeneratorOpen<>
  ContinuousUniformGenerator;

  ContinuousUniformGenerator::DiscreteUniformGenerator generator;
  ContinuousUniformGenerator random(&generator);

  // Determine an appropriate magnitude for the velocities.
  Float velMag = std::pow(_Order::Dimension *
                          (1. - 1. / order->particles.size()) *
                          State<Float, _Order::Dimension>::temperature(),
                          1. / _Order::Dimension);
  // Set the velocities and compute the sum.
  Point vSum = ext::filled_array<Point>(0);
  for (std::size_t i = 0; i != order->particles.size(); ++i) {
    for (std::size_t j = 0; j != _Order::Dimension; ++j) {
      order->particles[i].rv[j] = random() * velMag;
    }
    vSum += order->particles[i].rv;
  }
  // Correct so that the accumulated momenta are zero.
  for (std::size_t i = 0; i != order->particles.size(); ++i) {
    order->particles[i].rv -= vSum / Float(order->particles.size());
  }
}


template<typename _Order>
inline
void
initAccels(_Order* order)
{
  for (std::size_t i = 0; i != order->particles.size(); ++i) {
    order->particles[i].ra = ext::filled_array<typename _Order::Point>(0);
  }
}


template<typename _Float, std::size_t _Dimension>
void
run(const std::size_t latticeExtent, const _Float paddingFraction,
    const std::size_t numSteps)
{
  typedef typename particle::TemplatedTypes<_Float, _Dimension>::Point Point;
  typedef Mol<_Float, _Dimension> Particle;
  typedef particle::PeriodicTraits<Particle, GetPosition<_Float, _Dimension>,
          SetPosition<_Float, _Dimension>,
          _Dimension, _Float> Traits;
  typedef particle::MortonOrder<Traits> MortonOrder;

#if defined(NEIGHBORS_VERLET)
  typedef particle::VerletLists<MortonOrder> Neighbors;
#elif defined(NEIGHBORS_ADJACENT)
  typedef particle::AdjacentMask<MortonOrder> Neighbors;
#elif defined(NEIGHBORS_UNION_MASK)
  typedef particle::UnionMask<MortonOrder> Neighbors;
#else
#error Bad value for neighbors method.
#endif

  // The particles are initially placed at the centers of voxels.
  const std::array<std::size_t, _Dimension> latticeExtents =
    ext::filled_array<std::array<std::size_t, _Dimension> >(latticeExtent);
  // The lengths of the sides of the Cartesian domain.
  const Point lengths =
    _Float(1) / std::pow(State<_Float, _Dimension>::density(),
                         _Float(1) / _Dimension) *
    ext::convert_array<_Float>(latticeExtents);

  const geom::BBox<_Float, _Dimension> Domain =
  {ext::filled_array<Point>(0), lengths};
  MortonOrder mortonOrder(Domain,
                          State<_Float, _Dimension>::interactionDistance(),
                          State<_Float, _Dimension>::interactionDistance() *
                          paddingFraction);
  Neighbors neighbors(mortonOrder);

  State<_Float, _Dimension> state;
  setupJob(&mortonOrder, &state, latticeExtents);

  ads::Timer timer;
  double timeMoveParticles = 0;

  for (std::size_t n = 0; n != numSteps; ++n) {
    // Repair if necessary.
    mortonOrder.repair();
    // Find the neighbors.
    neighbors.findLocalNeighbors();

    timer.tic();
    singleStep(&mortonOrder, neighbors, &state);
    timeMoveParticles += timer.toc();
  }
  std::cout << "_Dimension = " << _Dimension
            << ", # particles = " << mortonOrder.particles.size() << '\n';
  mortonOrder.printPerformanceInfo(std::cout);
  neighbors.printPerformanceInfo(std::cout);
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
      << " [-t=T] [-h] [-l=L] [-f=F] [-s=S]\n"
      << "-t: The number of threads.\n"
      << "-h: Print this help message and exit.\n"
      << "-l: The lattice extent for the initial distribution of particles. The default is 10.\n"
      << "-f: The padding is this fraction of the interaction distance.\n"
      << "-s: The number of steps.\n";
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

  std::size_t latticeExtent = 10;
  parser.getOption('l', &latticeExtent);
  assert(latticeExtent > 0);

  float paddingFraction = std::numeric_limits<float>::quiet_NaN();
  parser.getOption('f', &paddingFraction);

  std::size_t numSteps = 1000;
  parser.getOption('s', &numSteps);
  assert(numSteps > 0);

  run<float, 3>(latticeExtent, paddingFraction, numSteps);

  return 0;
}
