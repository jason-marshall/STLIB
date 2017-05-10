// -*- C++ -*-

#if !defined(__sfc_LinearOrthantTrie_h__)
#define __sfc_LinearOrthantTrie_h__

/**
  \file
  \brief A linear orthant trie with contiguous storage.
*/

#include "stlib/sfc/AdaptiveCells.h"

#include <unordered_set>

namespace stlib
{
namespace sfc
{

/// A linear orthant trie with contiguous storage.
template<typename _Traits, typename _Cell, bool _StoreDel>
class LinearOrthantTrie :
    public OrderedCells<_Traits, _Cell, _StoreDel, BlockCode>
{
  //
  // Friends.
  //

  template<typename Traits_, typename Cell_, bool StoreObjectDelimiters_>
  friend
  std::ostream&
  operator<<
  (std::ostream& out,
   LinearOrthantTrie<Traits_, Cell_, StoreObjectDelimiters_> const& x);

  //
  // Types.
  //
private:

  /// The base class for ordered cells.
  typedef OrderedCells<_Traits, _Cell, _StoreDel, BlockCode> Base;

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
  using Base::_objectDelimiters;
  
  /// The index of the next non-overlapping cell.
  /** Note that the length is one more than the number of cells due to
    a guard element. */
  std::vector<std::size_t> _next;

  //--------------------------------------------------------------------------
  /** \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  /// The default constructor results in invalid member data.
  LinearOrthantTrie() :
    Base(),
    _next(1, std::size_t(-1))
  {
  }

  /// Construct from the domain and the number of levels of refinement.
  /** The lengths must be positive, but need not be the same. However, for
    most applications, the lengths will be the same as this maximizes the
    compactness of the cells. */
  LinearOrthantTrie(const Point& lowerCorner, const Point& lengths,
                    const std::size_t numLevels) :
    Base(lowerCorner, lengths, numLevels),
    _next(1, std::size_t(-1))
  {
  }

  /// Construct from the class that is used to order cells.
  /** This, in effect, specifies the domain and the number of levels of
    refinement. The list of cells will be empty. */
  LinearOrthantTrie(Grid const& grid) :
    Base(grid),
    _next(1, std::size_t(-1))
  {
  }

  /// Construct from a tight bounding box and a minimum allowed cell length.
  /** The domain will be expanded so that truncation errors will not result
    in points that lie outside of the domain of the trie, and then it
    will be extended so that the lengths in each dimension are the same.
    The cell length at the maximum level of refinement will not be less
    than the specifed minimum length. */
  LinearOrthantTrie(const BBox& tbb, const Float minCellLength = 0) :
    Base(tbb, minCellLength),
    _next(1, std::size_t(-1))
  {
  }

  /// Construct from multi-level cells.
  template<bool _Sod>
  LinearOrthantTrie(AdaptiveCells<_Traits, _Cell, _Sod> const&
                    adaptiveCells);

  /// Construct from uniform cells.
  template<bool _Sod>
  LinearOrthantTrie(UniformCells<_Traits, _Cell, _Sod> const& cellsUniform);

  /// Unserialize the cell data.
  /** We unserialize the cell data and then build the next links. This way we
   can directly use the capabilities in OrderedCells.
   \note There is no need to define a serialize() function here. We just
   inherit the capability from OrderedCells. */
  void
  unserialize(const std::vector<unsigned char>& buffer)
  {
    Base::unserialize(buffer);
    _linkNext();
  }

  //@}
  //--------------------------------------------------------------------------
  /// \name Accessors.
  //@{
public:

  /// Return true if the specified cell is a leaf.
  bool
  isLeaf(const std::size_t i) const
  {
    return ! isInternal(i);
  }

  /// Return true if the specified cell is internal.
  /** Internal nodes are followed immediately by a child. */
  bool
  isInternal(const std::size_t i) const
  {
    // Note that for a trie with zero levels of refinement, the lone cell
    // is a leaf. We have to check this case as it would be an error to 
    // access the parent.
    return Base::numLevels() != 0 &&
      _codes[i] == _grid.parent(_codes[i + 1]);
  }

  /// Return the level of the specified cell.
  std::size_t
  level(const std::size_t i) const
  {
    return _grid.level(_codes[i]);
  }

  /// Return the index of the next non-overlapping cell.
  std::size_t
  next(std::size_t const n) const
  {
    return _next[n];
  };

  /// Return a constant reference to the next non-overlapping cell.
  const CellRep&
  nextCell(std::size_t const n) const
  {
    return _cells[_next[n]];
  };

  /// Get the children of the specified cell.
  void
  getChildren(std::size_t i, Siblings<Dimension>* children) const
  {
    // Note that this function is defined in the class because the Intel 
    // compiler can't match the the function definition to its declaration.
#ifdef STLIB_DEBUG
    assert(isInternal(i));
#endif
    // The end code is the location for the next cell at the parent level.
    const Code endCode = _grid.location
      (_grid.next(_codes[i]));
    children->clear();
    // Note that the first child immediately follows the parent.
    for (++i; _codes[i] < endCode; i = _next[i]) {
      children->push_back(i);
    }
  }

  /// Count the number of leaves.
  std::size_t
  countLeaves() const;

  /// Return the required storage for the data structure (in bytes).
  std::size_t
  storage() const
  {
    return Base::storage() + _next.size() * sizeof(std::size_t);
  }

  /// Calculate the highest level for the cells.
  std::size_t
  calculateHighestLevel() const;

  /// Calculate the largest cell size for the leaves.
  /** \note The cell type must have the \c size() member function. */
  std::size_t
  calculateMaxLeafSize() const;

  /// Calculating object count statistics for the cells is not implemented.
  /** There's no reason I could not do this for the leaves, I'm just making
    sure that the function from the base class is not called, as it would
    give erroneous results. */
  void
  calculateObjectCountStatistics(std::size_t* min, std::size_t* max,
                                 Float* mean) const = delete;
  
  //@}
  //--------------------------------------------------------------------------
  /// \name Operations on the cells.
  //@{
public:

  /// Generate codes, sort the objects, and set the cells.
  /** This places objects in cells at the highest level of refinement. */
  template<typename _Object>
  void
  buildCells(std::vector<_Object>* objects, OrderedObjects* orderedObjects = 0)
  {
    Base::buildCells(objects, orderedObjects);
    _insertInternal();
  }

  /// Generate codes, refine cells while sorting the objects, and set the cells.
  /** 
    \param objects The vector of objects will be sorted.
    \param maxElementsPerCell The maximum number of elements per cell (except
    for cells at the highest level of refinement.
    \param orderedObjects If the pointer is not null, the original order 
    of the objects will be stored.
  */
  template<typename _Object>
  void
  buildCells(std::vector<_Object>* objects, std::size_t maxElementsPerCell,
             OrderedObjects* orderedObjects = 0);

  /// Erase the cells.
  void
  clear();

  /// Shrink the capacity to match the size.
  void
  shrink_to_fit();

  /// Check that the data structure is valid, including internal nodes.
  void
  checkValidity() const;

  /// Return true if the other is equal.
  bool
  operator==(const LinearOrthantTrie& other) const
  {
    return Base::operator==(other) && _next == other._next;
  }

private:

  void
  _checkObjectDelimiters() const;

  /// Add internal nodes to the trie. Set values for the cell's next indices.
  void
  _insertInternal();

  /// Sort the cells according to the codes. Used in _insertInternal().
  void
  _sortCells();

  /// Erase internal nodes from the trie.
  void
  _eraseInternal();

  /// Set the next links for each cell.
  void
  _linkNext();

  /// Merge cell information from children into parents.
  void
  _mergeToParents();

  //@}
};


/// Build a LinearOrthantTrie for the objects.
/**
   \relates LinearOrthantTrie

   \param objects The vector of objects will be reordered using the 
   space-filling curve.
   \param maxObjectsPerCell The maximum number of objects allowed in a cell
   (except perhaps for a cell at the highest level of refinement).
*/
template<typename _LinearOrthantTrie, typename _Object>
inline
_LinearOrthantTrie
linearOrthantTrie(std::vector<_Object>* objects,
                  std::size_t const maxObjectsPerCell = 64)
{
  typedef typename _LinearOrthantTrie::Traits Traits;
  typedef typename _LinearOrthantTrie::Cell Cell;
  bool constexpr StoreDel = _LinearOrthantTrie::StoreDel;
  typedef AdaptiveCells<Traits, Cell, StoreDel> CML;
  return _LinearOrthantTrie(adaptiveCells<CML>(objects, maxObjectsPerCell));
}


/// Build a LinearOrthantTrie for the objects. Record the original object order.
/**
   \relates LinearOrthantTrie

   \param objects The vector of objects will be reordered using the 
   space-filling curve.
   \param orderedObjects Data structure that may be used to restore the 
   original order of the objects.
   \param maxObjectsPerCell The maximum number of objects allowed in a cell
   (except perhaps for a cell at the highest level of refinement).
*/
template<typename _LinearOrthantTrie, typename _Object>
inline
_LinearOrthantTrie
linearOrthantTrie(std::vector<_Object>* objects,
                  OrderedObjects* orderedObjects,
                  std::size_t const maxObjectsPerCell = 64)
{
  assert(orderedObjects != nullptr);
  typedef typename _LinearOrthantTrie::Traits Traits;
  typedef typename _LinearOrthantTrie::Cell Cell;
  bool constexpr StoreDel = _LinearOrthantTrie::StoreDel;
  typedef AdaptiveCells<Traits, Cell, StoreDel> CML;
  return _LinearOrthantTrie(adaptiveCells<CML>(objects, orderedObjects,
                                                 maxObjectsPerCell));
}


/// Write information about the ordering and the cells.
template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::ostream&
operator<<
(std::ostream& out,
 LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& x)
{
  out << x._grid;
  for (std::size_t i = 0; i != x._codes.size(); ++i) {
    out << i
        << ", lev = " << x._grid.level(x._codes[i])
        << ", loc = " << std::hex
        << (x._grid.location(x._codes[i]) >> x._grid.levelBits())
        << std::dec;
    if (_StoreDel) {
      out << ", delim = " << x._objectDelimiters[i];
    }
    if (LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
        AreStoringCells) {
      out << ", cell = " << x._cells[i];
    }
    out << '\n';
  }
  return out;
}


} // namespace sfc
} // namespace stlib

#define __sfc_LinearOrthantTrie_tcc__
#include "stlib/sfc/LinearOrthantTrie.tcc"
#undef __sfc_LinearOrthantTrie_tcc__

#endif
