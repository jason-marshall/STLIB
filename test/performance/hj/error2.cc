// -*- C++ -*-

/*!
  \file performance/hj/error2.cc
  \brief Measure error in 2-D.
*/

/*!
  \page hj_error2 Error in 2-D

  The \e error program measures the error in solving the homogeneous
  eikonal equation to compute distance.  It can use any of the following
  methods:
  - f: Fast marching using a binary heap,
  - d: fast marching using a binary heap with Dynamic keys,
  - m: Marching with a correctness criterion,
  - c: marching with a correctness criterion using a Cell array,
  - b: Breadth-first search,
  - s: Sorted grid points.
  Usage:

  \verbatim
  ./error <method> <stencil> <order> <x size> <y size> <dx> <input> [exact] [solution] [error] \endverbatim

  The method must be either \c f, \c d, \c m, \c c \c b or \c s.  The
  stencil is either \c a or \c d (Adjacent or adjacent-Diagonal).  The
  order may be either \c 1 or \c 2 for first or second-order accurate
  difference schemes.  Next specify the size of the grid by giving the
  number of grid points in the x and y directions.  \c dx is the grid
  spacing.  \c input is the file containing the initial condition.
  See the files on the \c data directory for examples.
  \c exact is the optional output file for the exact solution.
  \c solution is the optional output file for the solution.
  \c error is the optional output file for the error.

  For example, consider computing the distance from a point in the
  lower left corner of the grid using the fast marching method and a
  second-order accurate, adjacent difference scheme on a \f$ 101
  \times 101 \f$ grid.  (If the domain is \f$ [0..1]^2 \f$ then \c dx
  is \f$ 0.01 \f$.)  The shell command is:

  \verbatim
  ./error f a 2 101 101 0.01 data/Corner.dat \endverbatim

  Next consider computing the distance with the marching with a
  correctness criterion algorithm.  We use a first-order accurate,
  adjacent-diagonal difference scheme.  We save the exact solution, the
  computed solution, and the error in the computed solution.
  The shell command is:

  \verbatim
  ./error f a 1 101 101 0.01 data/Corner.dat exact.out solution.out error.out \endverbatim

 */

#include "stlib/hj/hj.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <cassert>

using namespace stlib;

//
// Function declarations.
//

//! Compute the error in the numerical solution.
template <class DifferenceScheme>
void
error(hj::Grid<2, double, DifferenceScheme>& grid);

//
// Global variables.
//
static std::ifstream input_stream;
static std::ofstream exact_stream;
static std::ofstream solution_stream;
static std::ofstream error_stream;
static bool write_exact;
static bool write_solution;
static bool write_error;

//! Main.
int
main(int argc, char* argv[])
{
  if (argc < 8) {
    std::cerr << "Bad arguments.  Usage:" << '\n'
              << argv[0]
              << " <method> <stencil> <order> <x size> <y size> <dx>"
              << " <input> [exact] [solution] [error]"
              << '\n'
              << "<method> = f | d | m | c | b | s" << '\n'
              << "  f = Fast marching, binary heap" << '\n'
              << "  d = Fast marching, binary heap, Dynamic keys" << '\n'
              << "  m = Marching with a correctness criterion" << '\n'
              << "  c = MCC with a Cell array" << '\n'
              << "  b = Breadth-first search" << '\n'
              << "  s = Sorted grid points" << '\n'
              << "<stencil> = a | d" << '\n'
              << "<order> = 1 | 2" << '\n'
              << "<input> is the file containing the initial condition"
              << '\n'
              << "[exact] is the output file for the exact solution."
              << '\n'
              << "[solution] is the output file for the solution."
              << '\n'
              << "[error] is the output file for the error."
              << '\n';
    exit(1);
  }

  //
  // Get the command line options.
  //

  --argc;
  ++argv;
  char method;
  {
    std::istringstream str(*argv);
    str >> method;
    if (!(method == 'f' || method == 'd' || method == 'm' || method == 'c' ||
          method == 'b' || method == 's')) {
      std::cerr << "Bad method." << '\n';
      exit(1);
    }
  }

  --argc;
  ++argv;
  char stencil;
  {
    std::istringstream str(*argv);
    str >> stencil;
    if (!(stencil == 'a' || stencil == 'd')) {
      std::cerr << "Bad stencil." << '\n';
      exit(1);
    }
  }

  --argc;
  ++argv;
  std::size_t order;
  {
    std::istringstream str(*argv);
    str >> order;
    if (!(order == 1 || order == 2)) {
      std::cerr << "Bad order." << '\n';
      exit(1);
    }
  }

  --argc;
  ++argv;
  std::size_t x_size;
  {
    std::istringstream str(*argv);
    str >> x_size;
    if (x_size <= 0) {
      std::cerr << "Bad x size." << '\n';
      exit(1);
    }
  }

  --argc;
  ++argv;
  std::size_t y_size;
  {
    std::istringstream str(*argv);
    str >> y_size;
    if (y_size <= 0) {
      std::cerr << "Bad y size." << '\n';
      exit(1);
    }
  }

  --argc;
  ++argv;
  double dx;
  {
    std::istringstream str(*argv);
    str >> dx;
    if (dx <= 0) {
      std::cerr << "Bad dx." << '\n';
      exit(1);
    }
  }

  --argc;
  ++argv;
  input_stream.open(*argv);
  if (!input_stream) {
    std::cerr << "Bad input file." << '\n';
    exit(1);
  }

  --argc;
  ++argv;
  write_exact = false;
  if (argc > 0) {
    write_exact = true;
    exact_stream.open(*argv);
    if (!exact_stream) {
      std::cerr << "Bad exact solution file." << '\n';
      exit(1);
    }
  }

  --argc;
  ++argv;
  write_solution = false;
  if (argc > 0) {
    write_solution = true;
    solution_stream.open(*argv);
    if (!solution_stream) {
      std::cerr << "Bad solution file." << '\n';
      exit(1);
    }
  }

  --argc;
  ++argv;
  write_error = false;
  if (argc > 0) {
    write_error = true;
    error_stream.open(*argv);
    if (!error_stream) {
      std::cerr << "Bad error file." << '\n';
      exit(1);
    }
  }

  std::cout << "Grid size = " << x_size << " x " << y_size << ".  dx = " << dx
            << '\n';


  typedef container::MultiArray<double, 2> MultiArray;
  typedef MultiArray::SizeList SizeList;
  MultiArray solution(SizeList{{x_size, y_size}});

  if (method == 'f') {
    if (stencil == 'a') {
      if (order == 1) {
        std::cout << "Computing distance with the FM_BH method."
                  << '\n'
                  << "First-order, adjacent difference scheme."
                  << '\n';
        hj::GridFM_BH < 2, double, hj::DiffSchemeAdj < 2, double,
        hj::DistanceAdj1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the FM_BH method."
                  << '\n'
                  << "Second-order, adjacent difference scheme."
                  << '\n';
        hj::GridFM_BH < 2, double, hj::DiffSchemeAdj < 2, double,
        hj::DistanceAdj2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
    else { // stencil == 'd'
      if (order == 1) {
        std::cout << "Computing distance with the FM_BH method."
                  << '\n'
                  << "First-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridFM_BH < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the FM_BH method."
                  << '\n'
                  << "Second-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridFM_BH < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
  }
  else if (method == 'd') {
    if (stencil == 'a') {
      if (order == 1) {
        std::cout << "Computing distance with the FM_BHDK method."
                  << '\n'
                  << "First-order, adjacent difference scheme."
                  << '\n';
        hj::GridFM_BHDK < 2, double, hj::DiffSchemeAdj < 2, double,
        hj::DistanceAdj1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the FM_BHDK method."
                  << '\n'
                  << "Second-order, adjacent difference scheme."
                  << '\n';
        hj::GridFM_BHDK < 2, double, hj::DiffSchemeAdj < 2, double,
        hj::DistanceAdj2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
    else { // stencil == 'd'
      if (order == 1) {
        std::cout << "Computing distance with the FM_BHDK method."
                  << '\n'
                  << "First-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridFM_BHDK < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the FM_BHDK method."
                  << '\n'
                  << "Second-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridFM_BHDK < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
  }
  else if (method == 'm') {
    if (stencil == 'a') {
      if (order == 1) {
        std::cout << "Computing distance with the MCC algorithm."
                  << '\n'
                  << "First-order, adjacent difference scheme."
                  << '\n';
        hj::GridMCC < 2, double, hj::DiffSchemeAdj < 2, double,
        hj::DistanceAdj1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the MCC algorithm."
                  << '\n'
                  << "Second-order, adjacent difference scheme."
                  << '\n';
        hj::GridMCC < 2, double, hj::DiffSchemeAdj < 2, double,
        hj::DistanceAdj2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
    else { // stencil == 'd'
      if (order == 1) {
        std::cout << "Computing distance with the MCC algorithm."
                  << '\n'
                  << "First-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridMCC < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the MCC algorithm."
                  << '\n'
                  << "Second-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridMCC < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
  }
  else if (method == 'c') {
    if (stencil == 'a') {
      std::cerr << "Cannot use adjacent scheme with MCC_CA." << '\n';
      exit(1);
    }
    else { // stencil == 'd'
      if (order == 1) {
        std::cout << "Computing distance with the MCC_CA algorithm."
                  << '\n'
                  << "First-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridMCC_CA < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the MCC_CA algorithm."
                  << '\n'
                  << "Second-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridMCC_CA < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
  }
  else if (method == 'b') {
    if (stencil == 'a') {
      if (order == 1) {
        std::cout << "Computing distance with the BFS method."
                  << '\n'
                  << "First-order, adjacent difference scheme."
                  << '\n';
        hj::GridBFS < 2, double, hj::DiffSchemeAdj < 2, double,
        hj::DistanceAdj1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the BFS method."
                  << '\n'
                  << "Second-order, adjacent difference scheme."
                  << '\n';
        hj::GridBFS < 2, double, hj::DiffSchemeAdj < 2, double,
        hj::DistanceAdj2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
    else { // stencil == 'd'
      if (order == 1) {
        std::cout << "Computing distance with the BFS method."
                  << '\n'
                  << "First-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridBFS < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the BFS method."
                  << '\n'
                  << "Second-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridBFS < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
  }
  else if (method == 's') {
    if (stencil == 'a') {
      if (order == 1) {
        std::cout << "Computing distance with the Sort method."
                  << '\n'
                  << "First-order, adjacent difference scheme."
                  << '\n';
        hj::GridSort < 2, double, hj::DiffSchemeAdj < 2, double,
        hj::DistanceAdj1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the Sort method."
                  << '\n'
                  << "Second-order, adjacent difference scheme."
                  << '\n';
        hj::GridSort < 2, double, hj::DiffSchemeAdj < 2, double,
        hj::DistanceAdj2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
    else { // stencil == 'd'
      if (order == 1) {
        std::cout << "Computing distance with the Sort method."
                  << '\n'
                  << "First-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridSort < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag1st<2, double> > >
        grid(solution, dx);
        error(grid);
      }
      else { // order == 2
        std::cout << "Computing distance with the Sort method."
                  << '\n'
                  << "Second-order, adjacent-diagonal difference scheme."
                  << '\n';
        hj::GridSort < 2, double, hj::DiffSchemeAdjDiag < 2, double,
        hj::DistanceAdjDiag2nd<2, double> > >
        grid(solution, dx);
        error(grid);
      }
    }
  }
  else {
    assert(false);
  }

  return 0;
}

//! 2-D Cartesian point.
typedef std::array<double, 2> Point;

//! Read the initial condition.
template <class DifferenceScheme>
void
read_initial_condition(hj::Grid<2, double, DifferenceScheme>& grid,
                       std::vector<Point>& sources)
{
  const std::size_t x_size = grid.solution().extents()[0];
  const std::size_t y_size = grid.solution().extents()[1];
  // The domain spanned by the grid.
  double x_min, y_min, x_max, y_max;
  input_stream >> x_min >> y_min >> x_max >> y_max;
  assert(input_stream && ! input_stream.eof());
  while (! input_stream.eof()) {
    double x, y;
    input_stream >> x >> y;
    if (! input_stream.eof()) {
      // Convert to grid coordinates.
      x = (x - x_min) / (x_max - x_min) * (x_size - 1);
      y = (y - y_min) / (y_max - y_min) * (y_size - 1);
      sources.push_back(Point{{x, y}});
    }
  }
  input_stream.close();
}


//! Initialize the grid.
template <class DifferenceScheme>
void
initialize(hj::Grid<2, double, DifferenceScheme>& grid,
           const std::vector<Point>& sources)
{
  // Initialize the grid for a solve.
  grid.initialize();

  // Set the inital condition
  for (std::vector<Point>::const_iterator i = sources.begin();
       i != sources.end(); ++i) {
    grid.add_source(*i);
  }
}


//! Special case for hj::GridSort.
template <class DifferenceScheme>
void
initialize(hj::GridSort<2, double, DifferenceScheme>& grid,
           const std::vector<Point>& sources)
{
  // Initialize the grid for a solve.
  grid.initialize();

  // Set the inital condition
  for (std::vector<Point>::const_iterator i = sources.begin();
       i != sources.end(); ++i) {
    grid.add_source((*i)[0], (*i)[1]);
  }

  // Pre-solve the problem.
  grid.pre_solve();

  // Re-set the initial condition.
  grid.initialize();
  for (std::vector<Point>::const_iterator i = sources.begin();
       i != sources.end(); ++i) {
    grid.add_source((*i)[0], (*i)[1]);
  }
}


template <class DifferenceScheme>
void
error(hj::Grid<2, double, DifferenceScheme>& grid)
{
  typedef container::MultiArray<double, 2> SolutionArray;
  typedef SolutionArray::SizeList SizeList;

  const std::size_t x_size = grid.solution().extents()[0];
  const std::size_t y_size = grid.solution().extents()[1];
  const double dx = grid.dx();

  //
  // Read the initial condition.
  //
  std::vector<Point> sources;
  read_initial_condition(grid, sources);

  //
  // Set the initial condition.
  //
  initialize(grid, sources);

  //
  // Compute the solution on the grid.
  //
  grid.solve();

  //
  // Compute the exact solution and the error.
  //
  SolutionArray solution(SizeList{{x_size, y_size}},
                         std::numeric_limits<double>::max());
  double l1 = 0, l2 = 0, linf = 0;
  {
    std::vector< Point >::const_iterator src;
    std::vector< Point >::const_iterator
    src_end = sources.end();
    double soln, err;
    for (std::size_t i = 0; i != x_size; ++i) {
      for (std::size_t j = 0; j != y_size; ++j) {
        soln = std::numeric_limits<double>::max();
        for (src = sources.begin(); src != src_end; ++src) {
          soln = std::min(soln,
                          std::sqrt(std::pow(dx * (i - (*src)[0]), 2) +
                                    std::pow(dx * (j - (*src)[1]), 2)));
        }
        solution(i, j) = soln;
        err = grid.solution()(i, j) - soln;
        l1 += std::abs(err);
        l2 += err * err;
        linf = std::max(linf, std::abs(err));
      }
    }
  }
  l1 /= (x_size * y_size);
  l2 = std::sqrt(l2) / (x_size * y_size);
  std::cout << l1 << '\n' << l2 << '\n' << linf << '\n';

  //
  // Write the exact solution.
  //
  if (write_exact) {
    for (std::size_t j = 0; j != y_size; ++j) {
      for (std::size_t i = 0; i != x_size - 1; ++i) {
        exact_stream << solution(i, j) << " ";
      }
      exact_stream << solution(x_size - 1, j) << '\n';
    }
    exact_stream.close();
  }

  //
  // Write the solution.
  //
  if (write_solution) {
    for (std::size_t j = 0; j != y_size; ++j) {
      for (std::size_t i = 0; i != x_size - 1; ++i) {
        solution_stream << grid.solution()(i, j) << " ";
      }
      solution_stream << grid.solution()(x_size - 1, j) << '\n';
    }
    solution_stream.close();
  }

  //
  // Write the error.
  //
  if (write_error) {
    for (std::size_t j = 0; j != y_size; ++j) {
      for (std::size_t i = 0; i != x_size - 1; ++i) {
        error_stream << grid.solution()(i, j) - solution(i, j)
                     << " ";
      }
      error_stream << (grid.solution()(x_size - 1, j)
                       - solution(x_size - 1, j)) << '\n';
    }
    error_stream.close();
  }
}
