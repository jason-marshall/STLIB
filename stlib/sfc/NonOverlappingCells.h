// -*- C++ -*-

#if !defined(__sfc_NonOverlappingCells_h__)
#define __sfc_NonOverlappingCells_h__

/*!
  \file
  \brief Base class for data structures that use ordered cells.
*/

#include "stlib/sfc/OrderedCells.h"
#include "stlib/sfc/Codes.h"

namespace stlib
{
namespace sfc
{

//! Base class for data structures that have non-overlapping cells.
template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
class NonOverlappingCells :
    public OrderedCells<_Traits, _Cell, _StoreDel, _Grid>
{
  //
  // Types.
  //
private:

  //! The base class.
  typedef OrderedCells<_Traits, _Cell, _StoreDel, _Grid> Base;
 
public:

  /// The class that defines the virtual grid geometry and orders cells.
  typedef typename Base::Grid Grid;
  //! The actual representation of a cell.
  typedef typename Base::CellRep CellRep;
  //! The cell type.
  typedef typename Base::Cell Cell;
  //! The unsigned integer type is used for coordinates and codes.
  typedef typename _Traits::Code Code;
  //! The floating-point number type.
  typedef typename _Traits::Float Float;
  //! A Cartesian point.
  typedef typename _Traits::Point Point;
  //! A bounding box.
  typedef typename _Traits::BBox BBox;

protected:

  //! The container for the cells.
  typedef typename Base::CellContainer CellContainer;
  // The container for the object delimiters.
  typedef typename Base::ObjectDelimitersContainer ObjectDelimitersContainer;
  
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Traits::Dimension;
  //! Whether we are storing cells.
  BOOST_STATIC_CONSTEXPR bool AreStoringCells = Base::AreStoringCells;

  //
  // Member data.
  //
protected:

  using Base::_codes;
  using Base::_cells;
  using Base::_grid;
  using Base::_objectDelimiters;
  
  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  /// The default constructor results in invalid member data.
  NonOverlappingCells() :
    Base()
  {
  }

  //! Construct from the domain and the number of levels of refinement.
  /*! The lengths must be positive, but need not be the same. However, for
   most applications, the lengths will be the same as this maximizes the
   compactness of the cells. */
  NonOverlappingCells(const Point& lowerCorner, const Point& lengths,
                      std::size_t numLevels) :
    Base(lowerCorner, lengths, numLevels)
  {
  }

  //! Construct from the class that is used to order cells.
  /*! This, in effect, specifies the domain and the number of levels of
    refinement. The list of cells will be empty. */
  NonOverlappingCells(const Grid& order) :
    Base(order)
  {
  }

  //! Construct from a tight bounding box and a minimum allowed cell length.
  /*! The domain will be expanded so that truncation errors will not result
    in points that lie outside of the domain of the trie, and then it
    will be extended so that the lengths in each dimension are the same.
    The cell length at the maximum level of refinement will not be less
    than the specifed minimum length. */
  NonOverlappingCells(const BBox& tbb, Float minCellLength) :
    Base(tbb, minCellLength)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Operations on the cells.
  //@{
public:

  //! Check that the data structure is valid.
  void
  checkValidity() const;

  //! Merge the contents of the other cells into this data structure.
  /*! \pre The number of levels of refinement must be the same. */
  NonOverlappingCells&
  operator+=(NonOverlappingCells const& other);

  //! Crop to reduce to the specified cells.
  /*! \note If you are storing object delimitrs, you will also need to update
    your container of objects. */
  void
  crop(std::vector<std::size_t> const& cells);
  
  //! Crop to reduce to the specified cells. Update the container of objects accordingly.
  template<typename _Object>
  void
  crop(std::vector<std::size_t> const& cells, std::vector<_Object>* objects);
  
protected:

  //! Merge cells that overlap.
  void
  _mergeCells();

  //! Merge the two lists of cells into this one.
  void
  _merge(NonOverlappingCells const& a, NonOverlappingCells const& b);

private:

  //! There is no need to crop object delimiters if we're not storing them.
  void
  _cropObjectDelimiters(std::vector<std::size_t> const& /*cellIndices*/,
                        std::false_type /*_StoreDel*/)
  {
  }

  //! Crop the object delimiters.
  void
  _cropObjectDelimiters(std::vector<std::size_t> const& cellIndices,
                        std::true_type /*_StoreDel*/);

  //! Check the object delimiters.
  void
  _checkObjectDelimiters() const;

  //! Convert the object delimiters to cell sizes.
  void
  _delimitersToSizes(ObjectDelimitersContainer* sizes) const;
  
  //! Convert the cell sizes to object delimiters.
  void
  _sizesToDelimiters(ObjectDelimitersContainer const& sizes);
  
  //@}
};


} // namespace sfc
} // namespace stlib

#define __sfc_NonOverlappingCells_tcc__
#include "stlib/sfc/NonOverlappingCells.tcc"
#undef __sfc_NonOverlappingCells_tcc__

#endif
