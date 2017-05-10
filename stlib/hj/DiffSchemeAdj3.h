// -*- C++ -*-

/*!
  \file DiffSchemeAdj3.h
  \brief A class that supports finite difference operations for a 3-D grid.

  Adjacent stencil.
*/

#if !defined(__hj_DiffSchemeAdj3_h__)
#error This file is an implementation detail of the class DiffSchemeAdj.
#endif

#include "stlib/ads/algorithm/min_max.h"

namespace stlib
{
namespace hj {

//! Adjacent difference scheme.  1st order.
/*!
  \param T is the number type.
  \param Equation represents the equation to be solved.  The equation must
  supply functions that perform the finite differencing in one and two
  adjacent directions.

  This class implements the labeling operations for adjacent difference
  schemes in the \c label_neighbors() member function.
*/
template<typename T, class Equation>
class DiffSchemeAdj<3, T, Equation> :
   public DiffScheme<3, T, Equation> {
private:

   typedef DiffScheme<3, T, Equation> Base;

public:

   //! The number type.
   typedef typename Base::Number Number;
   //! A multi-index.
   typedef typename Base::IndexList IndexList;
   //! An index range.
   typedef typename Base::Range Range;

private:

   //! The index ranges of the grid, not counting ghost points.
   using Base::_index_ranges;
   //! A reference for the solution array.
   using Base::_solution;
   //! The status array.
   using Base::_status;
   //! The equation.
   using Base::_equation;

private:

   //
   // Not implemented.
   //

   //! Default constructor not implemented.
   DiffSchemeAdj();
   //! Copy constructor not implemented.
   DiffSchemeAdj(const DiffSchemeAdj&);
   //! Assignment operator not implemented.
   DiffSchemeAdj&
   operator=(const DiffSchemeAdj&);

public:

   //
   // Constructors
   //

   //! Constructor.
   /*!
     \param index_ranges are the index ranges of the real grid points
     (not counting the ghost grid points, if there are any).
     \param solution is the solution array.
     \param dx is the grid spacing.
     \param is_concurrent indicates whether we are using the concurrent
     algorithm.
   */
   DiffSchemeAdj(const Range& index_ranges,
                 container::MultiArrayRef<Number, 3>& solution,
                 const Number dx, const bool is_concurrent) :
      Base(index_ranges, solution, dx, is_concurrent) {}

   //
   // Mathematical member functions.
   //

   //! Return true if the grid point has an unknown neighbor.
   /*!
     Return true if an adjacent neighbor in the solution grid has a value
     of infinity.
   */
   bool
   has_unknown_neighbor(const IndexList& i) const;

   //! Label the unknown neighbors of the specified grid point.
   /*!
     Label the 6 adjacent neighbors.
     Add the neighbors that were unlabeled to \c labeled.
   */
   template<class Container>
   void
   label_neighbors(Container& labeled, IndexList i);

private:

   //! Use the difference scheme to compute the solution at \c i.
   /*!
     The grid point \c i + \c di must be known.
     It will be used in all finite difference computations.
   */
   Number
   diff(const IndexList& i, const IndexList& di) const;

   //! Use two neighbors to compute the solution at (\c i,\c j,\c k).
   Number
   diff_adj_adj(const IndexList& i, const IndexList& di) const;

   //! Use three neighbors to compute the solution at (\c i,\c j,\c k).
   Number
   diff_adj_adj_adj(const IndexList& i, const IndexList& di) const;

   //! Label the specified grid point with the value.
   using Base::label;

   //! Return true if the grid point is labeled or unlabeled.
   using Base::is_labeled_or_unlabeled;
};

} // namespace hj
}

#define __hj_DiffSchemeAdj3_ipp__
#include "stlib/hj/DiffSchemeAdj3.ipp"
#undef __hj_DiffSchemeAdj3_ipp__
