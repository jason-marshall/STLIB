// -*- C++ -*-

#if !defined(__sfc_UniformCells_h__)
#define __sfc_UniformCells_h__

/**
  \file
  \brief Ordered cells of uniform size (that is, at a single level).
*/

#include "stlib/sfc/NonOverlappingCells.h"
#include "stlib/sfc/LocationCode.h"

namespace stlib
{
namespace sfc
{

/// Ordered cells of uniform size (that is, at a single level).
template<typename _Traits, typename _Cell, bool _StoreDel>
class UniformCells :
    public NonOverlappingCells<_Traits, _Cell, _StoreDel,
                               LocationCode>
{
  //
  // Types.
  //
private:

  /// The base class for ordered cells.
  typedef NonOverlappingCells<_Traits, _Cell, _StoreDel,
                              LocationCode> Base;
 
public:

  /// The class that defines the virtual grid geometry and orders cells.
  typedef typename Base::Grid Grid;
  //! The actual representation of a cell.
  typedef typename Base::CellRep CellRep;
  /// The cell type.
  typedef typename Base::Cell Cell;
  /// The unsigned integer type is used for coordinates and codes.
  typedef typename _Traits::Code Code;
  /// The floating-point number type.
  typedef typename _Traits::Float Float;
  /// A Cartesian point.
  typedef typename _Traits::Point Point;
  /// A bounding box.
  typedef typename _Traits::BBox BBox;

  //
  // Constants.
  //
public:

  /// The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Traits::Dimension;

  //
  // Member data.
  //
protected:

  using Base::_codes;
  using Base::_cells;
  using Base::_grid;
  
  //--------------------------------------------------------------------------
  /** \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  /// The default constructor results in invalid member data.
  UniformCells() :
    Base()
  {
  }

  /// Construct from the domain and the number of levels of refinement.
  /** The lengths must be positive, but need not be the same. However, for
   most applications, the lengths will be the same as this maximizes the
   compactness of the cells. */
  UniformCells(const Point& lowerCorner, const Point& lengths,
               const std::size_t numLevels) :
    Base(lowerCorner, lengths, numLevels)
  {
  }

  /// Construct from the class that is used to order cells.
  /** This, in effect, specifies the domain and the number of levels of
    refinement. The list of cells will be empty. */
  UniformCells(Grid const& grid) :
    Base(grid)
  {
  }

  /// Construct from a tight bounding box and a minimum allowed cell length.
  /** The domain will be expanded so that truncation errors will not result
    in points that lie outside of the domain of the trie, and then it
    will be extended so that the lengths in each dimension are the same.
    The cell length at the maximum level of refinement will not be less
    than the specifed minimum length. */
  UniformCells(const BBox& tbb, const Float minCellLength = 0) :
    Base(tbb, minCellLength)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  /// \name Operations on the cells.
  //@{
public:

  /// Coarsen the cells if necessary to set the number of levels of refinement.
  /** \note The specified number of levels of refinement must be no greater
   than the current value. */
  void
  setNumLevels(std::size_t levels);

  /// Coarsen the cells by one level.
  void
  coarsen();

  /// Coarsen as many levels as possible while keeping under the specified cell size.
  /** \return The number of levels that were coarsened. */
  std::size_t
  coarsen(std::size_t cellSize);

private:

  /// Return true if by coarsening we would not exceed the specified cell size.
  bool
  _shouldCoarsen(std::size_t cellSize) const;

  //@}
};

} // namespace sfc
} // namespace stlib

#define __sfc_UniformCells_tcc__
#include "stlib/sfc/UniformCells.tcc"
#undef __sfc_UniformCells_tcc__

#endif
