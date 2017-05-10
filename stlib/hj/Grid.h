// -*- C++ -*-

/*!
  \file hj/Grid.h
  \brief A base class for sequential and concurrent N-D grids.
*/

#if !defined(__hj_Grid_h__)
#define __hj_Grid_h__

#include "stlib/container/MultiArray.h"

#include <iostream>
#include <limits>

// For sqrt().
#include <cmath>
#include <cassert>

namespace stlib
{
namespace hj {

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! A base class for sequential and concurrent N-D grids.
/*!
  The Grid class implements the common functionality for the sequential
  and concurrent algorithms.  It stores the solution array
  and the finite difference scheme.  This class does not have any
  functionality which is specific to the solution method or the labeling
  scheme.

  \param T is the number type.
  \param DiffferenceScheme is the finite difference scheme.
*/
template<std::size_t N, typename T, class DifferenceScheme>
class Grid {
public:

   //! The number type.
   typedef T Number;
   //! An array reference of floating point numbers.
   typedef container::MultiArrayRef<Number, N> NumberArrayRef;
   //! A multi-index into the solution array.
   typedef typename NumberArrayRef::IndexList IndexList;
   //! A single index into the solution array.
   typedef typename NumberArrayRef::Index Index;
   //! A Cartesian point.
   typedef std::array<Number, N> Point;
   //! A handle to a value type stored in an array.
   typedef typename NumberArrayRef::iterator handle;
   //! A const handle to a value type stored in an array.
   typedef typename NumberArrayRef::const_iterator const_handle;

protected:

   //
   // Protected enumerations and typedefs
   //

   //! Index range.
   typedef typename NumberArrayRef::Range Range;

protected:

   //
   // Member data
   //

   //! The solution array.
   NumberArrayRef _solution;

   //! The index ranges of the solution, not counting ghost points.
   Range _ranges;

   //! The grid spacing.
   Number _dx;

   //! The finite difference scheme.
   DifferenceScheme _scheme;

private:

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   Grid();
   //! Copy constructor not implemented.
   Grid(const Grid&);
   //! Assignment operator not implemented.
   Grid&
   operator=(const Grid&);

public:

   //
   // Constructors
   //

   //! Construct from the solution array and the grid spacing
   /*!
     \param solution is the solution array.
     \param dx is the spacing between adjacent grid points.
     \param is_concurrent indicates if this grid is for the concurrent
     solver.
   */
   Grid(container::MultiArrayRef<Number, N>& solution, Number dx,
        bool is_concurrent = false);

   //! Trivial destructor.
   virtual
   ~Grid() {}

   //
   // Accesors
   //

   //! Return a const reference to the index ranges of the solution array.
   /*!
     Do not count the ghost points.
   */
   const Range&
   ranges() const {
      return _ranges;
   }

   //! Return a const reference to the solution array.
   const NumberArrayRef&
   solution() const {
      return _solution;
   }

   //! Return the grid spacing.
   Number
   dx() const {
      return _dx;
   }

   //! Return a const reference to the difference scheme.
   const DifferenceScheme&
   scheme() const {
      return _scheme;
   }

   //
   // Manipulators
   //

   //! Return a reference to the solution array.
   NumberArrayRef&
   solution() {
      return _solution;
   }

   //! Return a reference to the difference scheme.
   DifferenceScheme&
   scheme() {
      return _scheme;
   }

   //
   // Mathematical member functions
   //

   //! Solve the Hamilton-Jacobi equation.
   /*!
     This function must be defined in child classes.

     Find the solution for all grid points with solution less than or
     equal to \c max_solution.  If \c max_solution is zero, then the solution
     is determined for the entire grid.  The default value of \c max_solution
     is zero.
   */
   virtual
   void
   solve(Number max_solution = 0) = 0;

   //! Initialize the solution array and initialize the difference scheme.
   /*!
     Set the solution array to \c std::numeric_limits<T>::max().
   */
   void
   initialize();

   //! Read the initial condition and initialize the difference scheme.
   /*!
     All the points in the solution array that are not infinite are taken
     to be known.  Initialize the difference scheme from this information.

     \return The number of grid points with \c INITIAL status.
   */
   std::size_t
   set_unsigned_initial_condition();

   //! Read the initial condition with negative values.
   /*!
     Initialize the difference scheme.
     All the points in the solution array that are finite are taken
     to be known.  Set the status of the non-positive grid points to
     \c INITIAL and the status of the positive grid points to \c KNOWN.
     Then reverse the sign of the solution.

     \return The number of grid points with \c INITIAL status.
   */
   std::size_t
   set_negative_initial_condition();

   //! Reverse the sign of the known grid points.
   /*!
     After reversing the sign, set the status of finite, positive points
     to \c INITIAL.

     \return The number of grid points with \c INITIAL status.
   */
   std::size_t
   set_positive_initial_condition();

   //! Set the \c value at the (\c i,\c j,\c k) grid point.
   /*!
     Set the status to \c INITIAL in the difference scheme.
   */
   void
   set_initial(const IndexList& i, Number value);

   //! Add a source at the specified position.
   /*
     \param x The coordinates in index space.
     The source must be in the index range of the grid.
   */
   void
   add_source(const std::array<Number, 2>& x);

   //! Add a source at the specified position.
   /*
     \param x The coordinates in index space.
     The source must be in the index range of the grid.
   */
   void
   add_source(const std::array<Number, 3>& x);

   //
   // File I/O
   //

   //! Write statistics for the status and solution arrays.
   void
   print_statistics(std::ostream& out) const;

   //! Write solution array and data associated with the difference scheme.
   void
   put(std::ostream& out) const;

private:

   // CONTINUE: Find a better solution for this.
   //! Return the distance with coordinates given in index space.
   Number
   index_distance(const IndexList& i, const std::array<Number, 2>& x)
      const {
      return std::sqrt((i[0] - x[0]) *(i[0] - x[0]) + (i[1] - x[1]) *(i[1] - x[1]))
             * _dx;
   }

   // CONTINUE: Find a better solution for this.
   //! Return the distance with coordinates given in index space.
   Number
   index_distance(const IndexList& i, const std::array<Number, 3>& x)
      const {
      return std::sqrt((i[0] - x[0]) *(i[0] - x[0]) + (i[1] - x[1]) *(i[1] - x[1]) +
                       (i[2] - x[2]) *(i[2] - x[2])) * _dx;
   }

   //! Return true if the indices are in the grid.
   bool
   indices_in_grid(const IndexList& i) const {
      return isIn(_solution.range(), i);
   }

};

//
// File I/O
//

//! Write to a file stream.
template<std::size_t N, typename T, class DifferenceScheme>
std::ostream&
operator<<(std::ostream& out, const Grid<N, T, DifferenceScheme>& x);

} // namespace hj
}

#define __hj_Grid_ipp__
#include "stlib/hj/Grid.ipp"
#undef __hj_Grid_ipp__

#endif
