// -*- C++ -*-

#if !defined(__sfc_OrderedCells_h__)
#define __sfc_OrderedCells_h__

/**
  \file
  \brief Base class for data structures that use ordered cells.
*/

#include "stlib/sfc/Cell.h"
#include "stlib/sfc/CellFunctors.h"
#include "stlib/sfc/Codes.h"
#include "stlib/sfc/OrderedObjects.h"
#include "stlib/sfc/sortByCodes.h"

#include "stlib/ads/algorithm/countGroups.h"
#include "stlib/ext/pair.h"
#include "stlib/container/DummyVector.h"

namespace stlib
{
namespace sfc
{

/// Base class for data structures that use ordered cells.
/** This data structure holds a vector of codes and a vector of cells.
  \c _Cell is the cell data type. For example, one might use
  CellDelimiter or CellBBox. The cell data type must be default
  constructable, as well as having a copy constructor and an
  assignment operator. \c _Grid is a data structure that defines the
  geometry for the virtual grid. It is used for calculating
  codes. For example, one might use LocationCode or BlockCode.
*/
template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
class OrderedCells
{
  //
  // Friends.
  //
private:

  // We need this for the copy constructor.
  friend class OrderedCells<_Traits, _Cell, false, _Grid>;
  
  template<typename Traits_, typename Cell_, bool StoreDel_,
           template<typename> class Grid_>
  friend
  std::ostream&
  operator<<(std::ostream& out,
             OrderedCells<Traits_, Cell_, StoreDel_, Grid_> const& x);

  //
  // Constants and types.
  //
public:

  /// The traits.
  typedef _Traits Traits;
  /// The specified cell type.
  /** This type may be void. In that case the representation of the cell is
      a dummy type. */
  typedef _Cell Cell;
  /// Whether we are storing cells.
  BOOST_STATIC_CONSTEXPR bool AreStoringCells = ! std::is_void<_Cell>::value;
  /// The representation of the cell type.
  /** When the template parameter is void, a dummy type is used. */
  typedef typename
  std::conditional<AreStoringCells, _Cell, DummyCell>::type CellRep;
  /// The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Traits::Dimension;

  /// The class that defines the virtual grid geometry and orders cells.
  typedef _Grid<_Traits> Grid;
  /// Whether to store object delimiters.
  BOOST_STATIC_CONSTEXPR bool StoreDel = _StoreDel;
  /// The unsigned integer type is used for coordinates and codes.
  typedef typename _Traits::Code Code;
  /// The floating-point number type.
  typedef typename _Traits::Float Float;
  /// A Cartesian point.
  typedef typename _Traits::Point Point;
  /// A bounding box.
  typedef typename _Traits::BBox BBox;

  /// The maximum possible levels of refinement given the dimension and integer type.
  BOOST_STATIC_CONSTEXPR std::size_t MaxLevels = Grid::MaxLevels;

protected:

  /// The container for the cells.
  typedef typename
  std::conditional<AreStoringCells,
                   std::vector<CellRep>, container::DummyVector<CellRep> >::type
  CellContainer;

  // The container for the object delimiters.
  typedef typename
  std::conditional<_StoreDel, std::vector<std::size_t>,
                   container::DummyVector<std::size_t> >::type
  ObjectDelimitersContainer;
  
  //
  // Member data.
  //
protected:

  /// The codes for each cell.
  /** Note that the length is one more than the number of cells due to
    a guard element, which has a value that is greater than any valid code. */
  std::vector<Code> _codes;
  /// The vector of cells.
  /** Note that the length is one more than the number of valid cells due to
    a guard element, which is necessary for cells than hold delimiters. The 
    value at the guard element is an empty cell. If the data structure is
    built using associated objects, then the guard cell is built from an 
    empty range of objects. */
  CellContainer _cells;
  /// The class for ordering the cells.
  Grid _grid;
  /// The class that may store object delimiters for the cells.
  ObjectDelimitersContainer _objectDelimiters;

  //--------------------------------------------------------------------------
  /** \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
    The constructors are protected because this class should not be used
    directly. Use the subclasses instead.
  */
  //@{
protected:

  /// The default constructor results in invalid member data.
  OrderedCells();

  /// Construct from the domain and the number of levels of refinement.
  /** The lengths must be positive, but need not be the same. However, for
   most applications, the lengths will be the same as this maximizes the
   compactness of the cells. */
  OrderedCells(const Point& lowerCorner, const Point& lengths,
               std::size_t numLevels);

  /// Construct from the class that is used to order cells.
  /** This, in effect, specifies the domain and the number of levels of
    refinement. The list of cells will be empty. */
  OrderedCells(const Grid& order);

  /// Construct from a tight bounding box and a minimum allowed cell length.
  /** The domain will be expanded so that truncation errors will not result
    in points that lie outside of the domain of the trie, and then it
    will be extended so that the lengths in each dimension are the same.
    The cell length at the maximum level of refinement will not be less
    than the specifed minimum length. */
  OrderedCells(const BBox& tbb, Float minCellLength);

  /// Copy constructor.
  /** \note You can copy from a data structure that stores object delimiters
    to one that does not. However the reverse is not possible. */
  OrderedCells(OrderedCells<_Traits, _Cell, true, _Grid> const& other);

public:

  /// Return the storage (in bytes) for the serialized cells.
  std::size_t
  serializedSize() const;

  /// Serialize the cells.
  /** \note The domain information is not recorded. */
  void
  serialize(std::vector<unsigned char>* buffer) const;

  /// Serialize the cells.
  /** Write to the buffer, which must have adequate capacity. Return the end
    of the written data.
    \note The domain information is not recorded. */
  unsigned char*
  serialize(unsigned char* buffer) const;

  /// Unserialize the cell data.
  /** Return the position past the read portion of the buffer. */
  std::size_t
  unserialize(std::vector<unsigned char> const& buffer,
              std::size_t const pos = 0)
  {
    return std::distance(&buffer[pos], unserialize(&buffer[pos]));
  }

  /// Unserialize the cell data.
  /** Return the end of the portion of the buffer that was read. */
  unsigned char const*
  unserialize(unsigned char const* buffer);

  //@}
  //--------------------------------------------------------------------------
  /// \name Accessors.
  //@{
public:

  /// Return the lower corner of the Cartesian domain.
  Point const&
  lowerCorner() const
  {
    return _grid.lowerCorner();
  }

  /// The lengths of the Cartesian domain.
  Point const&
  lengths() const
  {
    return _grid.lengths();
  }

  /// Return the number of levels of refinement.
  std::size_t
  numLevels() const
  {
    return _grid.numLevels();
  }

  /// Return the number of cells. Do not count the guard cell.
  std::size_t
  size() const
  {
    return _codes.size() - 1;
  }

  /// Return true if the number of cells is zero.
  bool
  empty() const
  {
    return size() == 0;
  }

  /// Return the required storage for the data structure (in bytes).
  std::size_t
  storage() const
  {
    return _codes.size() * sizeof(Code) +
      _cells.size() * sizeof(CellRep) +
      _objectDelimiters.size() * sizeof(std::size_t);
  }

  /// Return a constant reference to the nth cell.
  CellRep const&
  operator[](std::size_t const n) const
  {
    static_assert(AreStoringCells, "To use this function, the cell type must "
                  "not be void.");
    return _cells[n];
  };

  /// Return the nth code.
  Code
  code(std::size_t const n) const
  {
    return _codes[n];
  };

  /// Return a const reference to the vector of codes.
  /** \note The sequence is terminated with the guard code. */
  std::vector<Code> const&
  codesWithGuard() const
  {
    return _codes;
  }

  /// Return a vector of codes for the sequence of cell indices.
  std::vector<Code>
  codes(std::vector<std::size_t> const& cells) const
  {
    std::vector<Code> result(cells.size());
    for (std::size_t i = 0; i != result.size(); ++i) {
      result[i] = code(cells[i]);
    }
    return result;
  }

  /// Return the nth object delimiter.
  /** \note This may only be used if you are storing the object delimiters. */
  std::size_t
  delimiter(std::size_t const n) const
  {
    static_assert(_StoreDel, "This function is only available "
                  "when storing object delimiters.");
    return _objectDelimiters[n];
  };

  /// Return a const reference to the class for ordering the cells.
  Grid const&
  grid() const
  {
    return _grid;
  }

  /// Convert codes to cell indices.
  /** \pre The input codes must be strictly ascending and be a subset of the
    codes for this data structure.
    \return The vector of cell indices. */
  std::vector<std::size_t>
  codesToCells(std::vector<Code> const& codes) const;
  
  //@}
  //--------------------------------------------------------------------------
  /// \name Operations on the cells.
  //@{
public:

  /// Return a reference to the nth cell.
  CellRep&
  operator[](std::size_t const n)
  {
    static_assert(AreStoringCells,
                  "To use this function, the cell type must not be void.");
    return _cells[n];
  };

  /// Generate codes, sort the objects, and set the cells.
  template<typename _Object>
  void
  buildCells(std::vector<_Object>* objects, OrderedObjects* orderedObjects = 0);

  /// Generate codes and set the cells.
  /** The objects must already be sorted. */
  template<typename _Object>
  void
  buildCells(std::vector<_Object> const& objects);

  /// Build using the code/index pairs and the sorted objects.
  /**
    \param codeIndexPairs The codes are SFC codes for the objects. The index
    records the original index of the object. The vector of pairs is sorted
    according to the codes.
    \param objects The vector of objects is sorted according to the SFC codes.
    \param orderedObjects If this pointer is not null, the original order
    of the objects will be recorded so that the objects or associated data
    may be restored to the original order at a later time.
  */
  template<typename _Object>
  void
  buildCells(std::vector<std::pair<Code, std::size_t> > const&
             codeIndexPairs,
             std::vector<_Object> const& objects,
             OrderedObjects* orderedObjects = 0);

  /// Erase the cells.
  void
  clear();

  /// Shrink the capacity of the codes and cells to match the size.
  void
  shrink_to_fit();

  /// Check that the data structure is valid.
  void
  checkValidity() const;

  /// Return true if the other is equal.
  bool
  operator==(const OrderedCells& other) const
  {
    return _codes == other._codes && _cells == other._cells &&
      _grid == other._grid && _objectDelimiters == other._objectDelimiters;
  }

  /// Calculate object count statistics for the cells.
  /** \note This may only be used if you are storing the object delimiters. */
  void
  calculateObjectCountStatistics(std::size_t* min, std::size_t* max,
                                 Float* mean) const;

protected:

  /// Copy the object delimiters if necessary.
  template<typename _Other>
  void
  _copyObjectDelimiters(_Other const& other)
  {
    _copyObjectDelimiters(other, std::integral_constant<bool,
                          _StoreDel>());
  }

  //@}

private:

  /// No need to copy the object delimiters.
  template<typename _Other>
  void
  _copyObjectDelimiters(_Other const& /*other*/,
                        std::false_type /*StoreObjectDelimiters*/)
  {
  }

  /// Copy the object delimiters.
  template<typename _Other>
  void
  _copyObjectDelimiters(_Other const& other,
                        std::true_type /*StoreObjectDelimiters*/)
  {
    _objectDelimiters.resize(other.size() + 1);
    for (std::size_t i = 0; i != _objectDelimiters.size(); ++i) {
      _objectDelimiters[i] = other.delimiter(i);
    }
  }
};


/// Write information about the ordering and the cells.
template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
std::ostream&
operator<<
(std::ostream& out,
 OrderedCells<_Traits, _Cell, _StoreDel, _Grid> const& x)
{
  out << x._grid;
  for (std::size_t i = 0; i != x._codes.size(); ++i) {
    out << i << ", code = " << x._codes[i];
    if (_StoreDel) {
      out << ", delim = " << x._objectDelimiters[i];
    }
    if (OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
        AreStoringCells) {
      out << ", cell = " << x._cells[i];
    }
    out << '\n';
  }
  return out;
}


} // namespace sfc
} // namespace stlib

#define __sfc_OrderedCells_tcc__
#include "stlib/sfc/OrderedCells.tcc"
#undef __sfc_OrderedCells_tcc__

#endif
