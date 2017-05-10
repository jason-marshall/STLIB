// -*- C++ -*-

/*!
  \file performance/hj/timer2.cc
  \brief Timer for computing distance in 2-D.
*/

/*!
  \page hj_timer2 Timer for computing distance in 2-D.

  The \e timer program measures the execution time of any of the six
  methods with either the homogeneous or inhomogeneous eikonal equation.
  - f: Fast marching using a binary heap,
  - d: fast marching using a binary heap with Dynamic keys,
  - m: Marching with a correctness criterion,
  - c: marching with a correctness criterion using a Cell array,
  - b: Breadth-first search,
  - s: Sorted grid points.
  Usage:

  \verbatim
  ./timer <method> <stencil> <order> <x size> <y size> <dx> <ratio> \endverbatim

  The method must be either \c f, \c d, \c m, \c c \c b or \c s.  The
  stencil is either \c a or \c d (Adjacent or adjacent-Diagonal).  The
  order may be either \c 1 or \c 2 for first or second order accurate
  difference schemes.  Next specify the size of the grid by giving the
  number of grid points in the x and y directions.  \c dx is the grid
  spacing.  \c ratio is the ratio of the highest to lowest propagation
  speed in the speed function \f$ f \f$.  If the ratio is unity, then
  the distance will be computed from a point in the center of the
  grid.  Otherwise the inhomogeneous eikonal equation is solved.  The
  speed function varies between \f$ 1 \f$ and \c ratio on the domain
  \f$ [0..1]^2 \f$.
  \f[
  f(x, y) = 1 + \frac{\mathrm{ratio} - 1}{2} (1 + \sin(6 \pi (x + y)))
  \f]

  For example, consider computing the distance from the center of the grid
  using the fast marching method and a second-order accurate, adjacent
  difference scheme on a \f$ 101 \times 101 \f$ grid.  (If the domain is
  \f$ [0..1]^2 \f$ then \c dx is \f$ 0.01 \f$.)  The shell command
  is:

  \verbatim
  ./timer f a 2 101 101 0.01 1 \endverbatim

  Next consider solving the eikonal equation with a speed ratio of 7
  with the marching with a correctness criterion algorithm.  We use a
  first-order accurate, adjacent-diagonal difference scheme.  The
  shell command is:

  \verbatim
  ./timer m d 1 101 101 0.01 7 \endverbatim

*/

#include "stlib/hj/hj.h"

#include "stlib/ads/timer.h"
#include "stlib/numerical/constants.h"

#include <iostream>
#include <sstream>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

//
// Function declarations.
//

//! Initialize to solve the eikonal equation.
template<class DifferenceScheme>
void
initialize(hj::Grid<2, float, DifferenceScheme>& grid, double ratio);

//! Initialize to solve the eikonal equation.
template<class DifferenceScheme>
void
initialize(hj::GridSort<2, float, DifferenceScheme>& grid, double ratio);

//! Initialize to compute the distance.
template<class DifferenceScheme>
void
initialize(hj::Grid<2, float, DifferenceScheme>& grid);

//! Initialize to compute the distance.
template<class DifferenceScheme>
void
initialize(hj::GridSort<2, float, DifferenceScheme>& grid);

//! Time the solve operation.
template<class DifferenceScheme>
void
time(hj::Grid<2, float, DifferenceScheme>& grid);


//! Main.
int
main(int argc, char* argv[])
{
  if (argc != 8) {
    std::cerr << "Bad arguments.  Usage:" << '\n'
              << argv[0]
              << " <method> <stencil> <order> <x size> <y size> <dx> <ratio>"
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
              << '\n';
    exit(1);
  }

  char method;
  {
    ++argv;
    std::istringstream str(*argv);
    str >> method;
    if (!(method == 'f' || method == 'd' || method == 'm' || method == 'c' ||
          method == 'b' || method == 's')) {
      std::cerr << "Bad method." << '\n';
      exit(1);
    }
  }

  char stencil;
  {
    ++argv;
    std::istringstream str(*argv);
    str >> stencil;
    if (!(stencil == 'a' || stencil == 'd')) {
      std::cerr << "Bad stencil." << '\n';
      exit(1);
    }
  }

  std::size_t order;
  {
    ++argv;
    std::istringstream str(*argv);
    str >> order;
    if (!(order == 1 || order == 2)) {
      std::cerr << "Bad order." << '\n';
      exit(1);
    }
  }

  std::size_t x_size;
  {
    ++argv;
    std::istringstream str(*argv);
    str >> x_size;
    if (x_size <= 0) {
      std::cerr << "Bad x size." << '\n';
      exit(1);
    }
  }

  std::size_t y_size;
  {
    ++argv;
    std::istringstream str(*argv);
    str >> y_size;
    if (y_size <= 0) {
      std::cerr << "Bad y size." << '\n';
      exit(1);
    }
  }

  double dx;
  {
    ++argv;
    std::istringstream str(*argv);
    str >> dx;
    if (dx <= 0) {
      std::cerr << "Bad dx." << '\n';
      exit(1);
    }
  }

  double ratio;
  {
    ++argv;
    std::istringstream str(*argv);
    str >> ratio;
    if (ratio < 1) {
      std::cerr << "Bad ratio." << '\n';
      exit(1);
    }
  }

  std::cout << "Grid size = " << x_size << " x " << y_size
            << ".  dx = " << dx
            << '\n';

  typedef container::MultiArray<float, 2> MultiArray;
  using SizeList = MultiArray::SizeList;
  MultiArray solution(SizeList{{x_size, y_size}});

  if (method == 'f') {
    if (stencil == 'a') {
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the FM_BH method."
                    << '\n'
                    << "First-order, adjacent difference scheme."
                    << '\n';
          hj::GridFM_BH < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::DistanceAdj1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the FM_BH method."
                    << '\n'
                    << "First-order, adjacent difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridFM_BH < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::EikonalAdj1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the FM_BH method."
                    << '\n'
                    << "Second-order, adjacent difference scheme."
                    << '\n';
          hj::GridFM_BH < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::DistanceAdj2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the FM_BH method."
                    << '\n'
                    << "Second-order, adjacent difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridFM_BH < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::EikonalAdj2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
    else { // stencil == 'd'
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the FM_BH method."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridFM_BH < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the FM_BH method."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridFM_BH < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the FM_BH method."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridFM_BH < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the FM_BH method."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridFM_BH < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
  }
  else if (method == 'd') {
    if (stencil == 'a') {
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the FM_BHDK method."
                    << '\n'
                    << "First-order, adjacent difference scheme."
                    << '\n';
          hj::GridFM_BHDK < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::DistanceAdj1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the FM_BHDK method."
                    << '\n'
                    << "First-order, adjacent difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridFM_BHDK < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::EikonalAdj1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the FM_BHDK method."
                    << '\n'
                    << "Second-order, adjacent difference scheme."
                    << '\n';
          hj::GridFM_BHDK < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::DistanceAdj2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the FM_BHDK method."
                    << '\n'
                    << "Second-order, adjacent difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridFM_BHDK < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::EikonalAdj2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
    else { // stencil == 'd'
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the FM_BHDK method."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridFM_BHDK < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the FM_BHDK method."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridFM_BHDK < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the FM_BHDK method."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridFM_BHDK < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the FM_BHDK method."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridFM_BHDK < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
  }
  else if (method == 'm') {
    if (stencil == 'a') {
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the MCC algorithm."
                    << '\n'
                    << "First-order, adjacent difference scheme."
                    << '\n';
          hj::GridMCC < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::DistanceAdj1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the MCC algorithm."
                    << '\n'
                    << "First-order, adjacent difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridMCC < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::EikonalAdj1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the MCC algorithm."
                    << '\n'
                    << "Second-order, adjacent difference scheme."
                    << '\n';
          hj::GridMCC < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::DistanceAdj2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the MCC algorithm."
                    << '\n'
                    << "Second-order, adjacent difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridMCC < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::EikonalAdj2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
    else { // stencil == 'd'
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the MCC algorithm."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridMCC < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the MCC algorithm."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridMCC < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the MCC algorithm."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridMCC < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the MCC algorithm."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridMCC < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
  }
  else if (method == 'c') {
    if (stencil == 'a') {
      std::cerr << "Cannot use an adjacent scheme with MCC_CA." << '\n';
      exit(1);
    }
    else { // stencil == 'd'
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the MCC_CA algorithm."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridMCC_CA < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the MCC_CA algorithm."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridMCC_CA < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the MCC_CA algorithm."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridMCC_CA < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the MCC_CA algorithm."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridMCC_CA < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
  }
  else if (method == 'b') {
    if (stencil == 'a') {
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the BFS method."
                    << '\n'
                    << "First-order, adjacent difference scheme."
                    << '\n';
          hj::GridBFS < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::DistanceAdj1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the BFS method."
                    << '\n'
                    << "First-order, adjacent difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridBFS < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::EikonalAdj1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the BFS method."
                    << '\n'
                    << "Second-order, adjacent difference scheme."
                    << '\n';
          hj::GridBFS < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::DistanceAdj2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the BFS method."
                    << '\n'
                    << "Second-order, adjacent difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridBFS < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::EikonalAdj2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
    else { // stencil == 'd'
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the BFS method."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridBFS < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the BFS method."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridBFS < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the BFS method."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridBFS < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the BFS method."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridBFS < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
  }
  else if (method == 's') {
    if (stencil == 'a') {
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the Sort method."
                    << '\n'
                    << "First-order, adjacent difference scheme."
                    << '\n';
          hj::GridSort < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::DistanceAdj1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the Sort method."
                    << '\n'
                    << "First-order, adjacent difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridSort < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::EikonalAdj1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the Sort method."
                    << '\n'
                    << "Second-order, adjacent difference scheme."
                    << '\n';
          hj::GridSort < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::DistanceAdj2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the Sort method."
                    << '\n'
                    << "Second-order, adjacent difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridSort < 2, float, hj::DiffSchemeAdj < 2, float,
          hj::EikonalAdj2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
    else { // stencil == 'd'
      if (order == 1) {
        if (ratio == 1) {
          std::cout << "Computing distance with the Sort method."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridSort < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the Sort method."
                    << '\n'
                    << "First-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridSort < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag1st<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
      else { // order == 2
        if (ratio == 1) {
          std::cout << "Computing distance with the Sort method."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n';
          hj::GridSort < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::DistanceAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid);
          time(grid);
        }
        else { // ratio != 1
          std::cout << "Solving the eikonal equation with the Sort method."
                    << '\n'
                    << "Second-order, adjacent-diagonal difference scheme."
                    << '\n'
                    << "Ratio = " << ratio
                    << '\n';
          hj::GridSort < 2, float, hj::DiffSchemeAdjDiag < 2, float,
          hj::EikonalAdjDiag2nd<2, float> > >
          grid(solution, dx);
          initialize(grid, ratio);
          time(grid);
        }
      }
    }
  }
  else {
    assert(false);
  }

  return 0;
}

template<class DifferenceScheme>
void
initialize(hj::Grid<2, float, DifferenceScheme>& grid, double ratio)
{
  typedef typename hj::Grid<2, float, DifferenceScheme>::IndexList IndexList;
  typedef typename hj::Grid<2, float, DifferenceScheme>::Index Index;

  grid.initialize();
  const std::size_t x_size = grid.solution().extents()[0];
  const std::size_t y_size = grid.solution().extents()[1];
  for (std::size_t i = 0; i != x_size; ++i) {
    for (std::size_t j = 0; j != y_size; ++j) {
      float speed =
        (ratio + 1) / 2 + (ratio - 1) / 2 *
        std::sin(6 * numerical::Constants<float>::Pi() *
                 (i / (x_size - 1.0) + j / (y_size - 1.0)));
      grid.scheme().equation().getInverseSpeed()(i, j) = 1.0 / speed;

    }
  }
  const IndexList i =
    ext::convert_array<Index>(grid.solution().extents() / std::size_t(2));
  grid.set_initial(i, 0);
}

template<class DifferenceScheme>
void
initialize(hj::GridSort<2, float, DifferenceScheme>& grid, double ratio)
{
  typedef typename hj::Grid<2, float, DifferenceScheme>::IndexList IndexList;
  typedef typename hj::Grid<2, float, DifferenceScheme>::Index Index;

  grid.initialize();
  const std::size_t x_size = grid.solution().extents()[0];
  const std::size_t y_size = grid.solution().extents()[1];
  for (std::size_t i = 0; i != x_size; ++i) {
    for (std::size_t j = 0; j != y_size; ++j) {
      float speed =
        (ratio + 1) / 2 + (ratio - 1) / 2 *
        std::sin(6 * numerical::Constants<float>::Pi() *
                 (i / (x_size - 1.0) + j / (y_size - 1.0)));
      grid.scheme().equation().getInverseSpeed()(i, j) = 1.0 / speed;
    }
  }
  const IndexList i =
    ext::convert_array<Index>(grid.solution().extents() / std::size_t(2));
  grid.set_initial(i, 0);
  grid.pre_solve();
  grid.initialize();
  grid.set_initial(i, 0);
}

template<class DifferenceScheme>
void
initialize(hj::Grid<2, float, DifferenceScheme>& grid)
{
  typedef typename hj::Grid<2, float, DifferenceScheme>::IndexList IndexList;
  typedef typename hj::Grid<2, float, DifferenceScheme>::Index Index;

  grid.initialize();
  const IndexList i =
    ext::convert_array<Index>(grid.solution().extents()) / Index(2);
  grid.set_initial(i, 0);
}

template<class DifferenceScheme>
void
initialize(hj::GridSort<2, float, DifferenceScheme>& grid)
{
  typedef typename hj::Grid<2, float, DifferenceScheme>::IndexList IndexList;
  typedef typename hj::Grid<2, float, DifferenceScheme>::Index Index;

  const IndexList i =
    ext::convert_array<Index>(grid.solution().extents()) / Index(2);
  grid.initialize();
  grid.set_initial(i, 0);
  grid.pre_solve();
  grid.initialize();
  grid.set_initial(i, 0);
}

template<class DifferenceScheme>
void
time(hj::Grid<2, float, DifferenceScheme>& grid)
{
  const std::size_t x_size = grid.solution().extents()[0];
  const std::size_t y_size = grid.solution().extents()[1];

  ads::Timer timer;
  timer.tic();
  grid.solve();
  double time = timer.toc();
  std::cout << "time = " << time << '\n'
            << "solution(" << x_size - 1 << "," << y_size - 1 << ") = "
            << grid.solution()(x_size - 1, y_size - 1)
            << '\n';

  std::cout << time << '\n';
}
