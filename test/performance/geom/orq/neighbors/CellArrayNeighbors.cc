// -*- C++ -*-

#include "stlib/geom/orq/CellArrayNeighbors.h"
#include "stlib/geom/orq/SpatialIndexMortonUniform.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer.h"
#include "stlib/numerical/random/uniform/DiscreteUniformGeneratorMt19937.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/numerical/random/normal/NormalGeneratorZigguratVoss.h"
#include "stlib/numerical/random/exponential/ExponentialGeneratorZiggurat.h"

#include <iterator>
#include <functional>

using namespace stlib;

//! A functor for dereferencing a handle.
/*!
  \param Handle is the handle type.
  \param Result is the return type of the dereferenced handle.  It has
  the default value: \c std::iterator_traits<Handle>::reference.  One could
  use the \c value_type as well.
*/
template<typename Handle,
         typename Result = typename std::iterator_traits<Handle>::reference>
struct Dereference :
    public std::unary_function<Handle, Result> {
  //! Return the object to which the handle \c x points.
  Result
  operator()(Handle x) const
  {
    return *x;
  }
};

std::string programName;

void
exitOnError()
{
  std::cerr
      << "Bad arguments.  Usage:\n"
      << programName << " numRecords searchRadius [-order=O] [-distribution=D]\n"
      << "Allowed values for the order are 'none', 'z', and 'morton' which result\n"
      << "in no ordering, sorting the records using the z-coordinate, and ordering\n"
      << "the records using the Morton spatial index, respectively.\n"
      << "The default ordering is 'none'.\n"
      << "Allowed values for the distribution are 'uniform', 'normal', and\n"
      << "'exponential'. The default is 'uniform'.\n"
      << "\nExiting...\n";
  exit(1);
}

typedef double Float;
const std::size_t Dimension = 3;
typedef std::array<Float, Dimension> Value;

bool
compare(const Value& x, const Value& y)
{
  return x[Dimension - 1] < y[Dimension - 1];
}

int
main(int argc, char* argv[])
{
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

  // Parse the program name and options.
  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();

  if (parser.getNumberOfArguments() != 2) {
    std::cerr << "Bad number of required arguments.\n"
              << "You gave the arguments:\n";
    parser.printArguments(std::cerr);
    exitOnError();
  }

  std::size_t numRecords;
  parser.getArgument(&numRecords);

  Float searchRadius;
  parser.getArgument(&searchRadius);

  const Float Length = std::pow(numRecords, 1. / 3.);

  typedef std::vector<Value> ValueContainer;
  typedef ValueContainer::const_iterator Record;
  typedef geom::CellArrayNeighbors<Float, Dimension, Record,
          Dereference<Record> > NS;

  std::string distribution("uniform");
  parser.getOption(std::string("distribution"), &distribution);

  // Random points.
  ValueContainer values(numRecords);
  typedef numerical::DiscreteUniformGeneratorMt19937 Generator;
  Generator generator;
  if (distribution == "uniform") {
    numerical::ContinuousUniformGeneratorOpen<Generator> random(&generator);
    for (std::size_t i = 0; i != values.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        values[i][j] = Length * random();
      }
    }
  }
  else if (distribution == "normal") {
    numerical::NormalGeneratorZigguratVoss<Generator> random(&generator);
    for (std::size_t i = 0; i != values.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        values[i][j] = Length * random();
      }
    }
  }
  else if (distribution == "exponential") {
    numerical::ExponentialGeneratorZiggurat<Generator> random(&generator);
    for (std::size_t i = 0; i != values.size(); ++i) {
      for (std::size_t j = 0; j != Dimension; ++j) {
        values[i][j] = Length * random();
      }
    }
  }
  else {
    std::cerr << "Unsupported point distribution: " << distribution << '\n';
    exitOnError();
  }

  // Print information on the bounding box for the points.
  {
    geom::BBox<Float, Dimension> box =
      geom::specificBBox<geom::BBox<Float, Dimension> >
      (values.begin(), values.end());
    std::cout << "The mean extent for the point domain is "
              << stlib::ext::sum(box.upper - box.lower) / Dimension << ".\n";
  }

  std::string order("none");
  parser.getOption(std::string("order"), &order);

  if (order == "z") {
    std::sort(values.begin(), values.end(), compare);
  }
  else if (order == "morton") {
    // Bound the points.
    geom::BBox<Float, Dimension> domain =
      geom::specificBBox<geom::BBox<Float, Dimension> >
      (values.begin(), values.end());
    // Build the spatial index functor.
    geom::SpatialIndexMortonUniform<Float, Dimension>
    f(domain, searchRadius);
    std::vector<std::size_t> indices(values.size());
    for (std::size_t i = 0; i != indices.size(); ++i) {
      indices[i] = f(values[i]);
    }
    ads::sortTogether(indices.begin(), indices.end(), values.begin(),
                      values.end());
  }
  else if (order != "none") {
    std::cerr << "Unsupported point ordering: " << order << '\n';
    exitOnError();
  }

  NS ns;
  std::size_t numNeighbors = 0;

  ads::Timer timer;
  timer.tic();
  ns.initialize(values.begin(), values.end());
  const double initializationTime = timer.toc();

  // The following command should work, generates a compilation error.
  // #pragma omp parallel for private(neighbors) reduction(+:numNeighbors)
  // Thus we split the pragmas.
  timer.tic();
  #pragma omp parallel
  {
    std::vector<Record> neighbors;
    #pragma omp for reduction(+:numNeighbors)
    for (int i = 0; i < int(values.size()); i++) {
      ns.neighborQuery(values[i], searchRadius, &neighbors);
      numNeighbors += neighbors.size();
    }
  }
  const double queryTime = timer.toc();

  std::cout << "Number of neighbors = " << numNeighbors << '\n'
            << "Average number of neighbors per record = "
            << double(numNeighbors) / values.size() << '\n'
            << "Initialization time = " << initializationTime << " seconds.\n"
            << "Query time = " << queryTime << " seconds.\n"
            << "Time per neighbor = " << queryTime / numNeighbors * 1e9
            << " nanoseconds\n";

  return 0;
}



