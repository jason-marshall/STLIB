// -*- C++ -*-

#if !defined(__sfc_AdaptiveCells_h__)
#define __sfc_AdaptiveCells_h__

/**
  \file
  \brief Ordered cells at multiple levels.
*/

#include "stlib/sfc/UniformCells.h"
#include "stlib/sfc/RefinementSort.h"
#include "stlib/sfc/Siblings.h"

namespace stlib
{
namespace sfc
{

/// Ordered cells at multiple levels.
template<typename _Traits, typename _Cell, bool _StoreDel>
class AdaptiveCells :
    public NonOverlappingCells<_Traits, _Cell, _StoreDel, BlockCode>
{
  //
  // Friends.
  //

  template<typename Traits_, typename Cell_, bool StoreDel_>
  friend
  std::ostream&
  operator<<(std::ostream& out,
             AdaptiveCells<Traits_, Cell_, StoreDel_> const& x);

  //
  // Types.
  //
private:

  /// The base class for ordered cells.
  typedef NonOverlappingCells<_Traits, _Cell, _StoreDel, BlockCode> Base;

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
  /// The default maximum number of objects per cell.
  /** This is used by some functions that build this data structure. */
  BOOST_STATIC_CONSTEXPR std::size_t DefaultMaxObjectsPerCell = 64;
  /// The default target number of cells per process.
  /** This is used by some functions that build data structures for distributed
      objects. This value was determined by testing the Daimler Aero geometry
      when computing unsigned distance with up to 512 processes. */
  BOOST_STATIC_CONSTEXPR std::size_t DefaultTargetCellsPerProcess = 100;

  //
  // Member data.
  //
protected:

  using Base::_codes;
  using Base::_cells;
  using Base::_grid;
  using Base::_objectDelimiters;
  
  //--------------------------------------------------------------------------
  /** \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  /// The default constructor results in invalid member data.
  AdaptiveCells() :
    Base()
  {
  }

  /// Construct from the domain and the number of levels of refinement.
  /** The lengths must be positive, but need not be the same. However, for
   most applications, the lengths will be the same as this maximizes the
   compactness of the cells. */
  AdaptiveCells(Point const& lowerCorner, Point const& lengths,
                  std::size_t const numLevels) :
    Base(lowerCorner, lengths, numLevels)
  {
  }

  /// Construct from the class that is used to order cells.
  /** This, in effect, specifies the domain and the number of levels of
    refinement. The list of cells will be empty. */
  AdaptiveCells(Grid const& grid) :
    Base(grid)
  {
  }

  /// Construct from a tight bounding box and a minimum allowed cell length.
  /** The domain will be expanded so that truncation errors will not result
    in points that lie outside of the domain of the trie, and then it
    will be extended so that the lengths in each dimension are the same.
    The cell length at the maximum level of refinement will not be less
    than the specifed minimum length. */
  AdaptiveCells(BBox const& tbb, Float const minCellLength = 0) :
    Base(tbb, minCellLength)
  {
  }

  /// Construct from uniform cells.
  /** The domain and the number of levels of refinement will be taken from
    the uniform cells. The cells are copied. Block codes for the multi-level
    cells are calculated. */
  template<bool _Sod>
  AdaptiveCells(UniformCells<_Traits, _Cell, _Sod> const& cellsUniform);

  /// Generate codes, refine cells while sorting the objects, and set the cells.
  /** 
    \param objects The vector of objects will be sorted.
    \param maxElementsPerCell The maximum number of elements per cell (except
    for cells at the highest level of refinement.
  */
  template<typename _Object>
  void
  buildCells(std::vector<_Object>* objects, std::size_t maxElementsPerCell,
             OrderedObjects* orderedObjects = 0);

  /// Build from a sorted sequence of cell code/size pairs.
  /** 
    \param codeSizePairs The vector of cell code/size pairs.
    \param maxElementsPerCell The maximum number of elements per cell (except
    for cells at the highest level of refinement.

    \note This function may only be used when you are not storing any
    information in the cells, that is, the cell type is \c void.
  */
  void
  buildCells(std::vector<std::pair<Code, std::size_t> > const&
             codeSizePairs);

  /// Build from the sequence of permissible cell codes.
  /**
    \param cellCodes The sequence of admissible cell codes. This is terminated
    with a guard code.
    \param objectCodes The codes for the (sorted) objects.
    \param objects The sorted sequence of objects.
   */
  template<typename _Object>
  void
  buildCells(std::vector<Code> const& cellCodes,
             std::vector<Code> const& objectCodes,
             std::vector<_Object> const& objects);

  // Because we defined a buildCells() function above, we need to bring the 
  // base class functions into scope.
  using Base::buildCells;

  //@}
  //--------------------------------------------------------------------------
  /// \name Operations on the cells.
  //@{
public:

  /// Calculate the highest level for the cells.
  std::size_t
  calculateHighestLevel() const;

  /// Set the number of levels of refinement to the highest level for the cells.
  /** This is a shrink-to-fit functionality for the maximum number of levels
   of refinement. */
  void
  setNumLevelsToFit();
  
  /// Apply all coarsening operations that do not require merging of cells.
  /** This operation is more efficient than other kinds of coarsening because
    it does not change the number of cells. Thus, the vector of cells does
    not need to be modified. Only the vector of codes is modified, and for that
    only the levels change. Because coarsening without merging is more 
    efficient, it is automatically applied first when you call 
    coarsenCellSize() or coarsenMaxCells().

    \return The number of coarsening operations. */
  std::size_t
  coarsenWithoutMerging();

  /// Coarsen using the supplied predicate.
  /** The predicate is a function of the level and a range of sibling cells.
    \return The number of coarsening operations. */
  template<typename _Predicate>
  std::size_t
  coarsen(_Predicate pred);

  /// Coarsen by increasing the leaf size until the number of cells is no more than specified.
  /** \param maxCells The maximum number of allowed cells. This must be
    positive.
    \return The number of coarsening operations. */
  std::size_t
  coarsenMaxCells(std::size_t maxCells);

  /// Coarsen using the supplied maximum cell size.
  /** \return The number of coarsening operations. */
  std::size_t
  coarsenCellSize(std::size_t cellSize);

private:

  /// Return true if there is a range of siblings that may be coarsened.
  /** \param begin The beginning of the range of siblings.
    \param end Either the end of the range of siblings or the first cell that
    is at a higher level.
    \param next The first position at which we could next try coarsening. */
  bool
  _getSiblings(std::size_t begin, std::size_t* end, std::size_t* next) const;

  /// Determine the smallest leaf size that will result in at least one coarsening operation.
  std::size_t
  _minSizeForCoarsening() const;

  /// Apply a sweep of coarsening operations that do not require merging of cells.
  /** \return The number of coarsening operations. */
  std::size_t
  _coarsenWithoutMergingSweep();

  /// Apply a sweep of coarsening using the supplied predicate.
  /** The predicate is a function of the level and a range of sibling cells.
    \return The number of coarsening operations. */
  template<typename _Predicate>
  std::size_t
  _coarsenSweep(_Predicate pred);

  /// Apply a sweep of coarsening using the supplied maximum cell size.
  std::size_t
  _coarsenSweepCellSize(std::size_t cellSize);

//@}
};


/// Build a AdaptiveCells for the objects.
/**
   \relates AdaptiveCells

   \param objects The vector of objects will be reordered using the 
   space-filling curve.
   \param maxObjectsPerCell The maximum number of objects allowed in a cell
   (except perhaps for a cell at the highest level of refinement).
*/
template<typename _AdaptiveCells, typename _Object>
_AdaptiveCells
adaptiveCells(std::vector<_Object>* objects,
              std::size_t maxObjectsPerCell =
              _AdaptiveCells::DefaultMaxObjectsPerCell);


/// Build a AdaptiveCells for the objects. Record the original object order.
/**
   \relates AdaptiveCells

   \param objects The vector of objects will be reordered using the 
   space-filling curve.
   \param orderedObjects Data structure that may be used to restore the 
   original order of the objects.
   \param maxObjectsPerCell The maximum number of objects allowed in a cell
   (except perhaps for a cell at the highest level of refinement).
*/
template<typename _AdaptiveCells, typename _Object>
_AdaptiveCells
adaptiveCells(std::vector<_Object>* objects,
              OrderedObjects* orderedObjects,
              std::size_t maxObjectsPerCell =
              _AdaptiveCells::DefaultMaxObjectsPerCell);


/// Determine an appropriate maximum number of objects per cell for MPI applications.
/**
\relates AdaptiveCells
\param numGlobalObjects The number distributed objects.
\param commSize The number of MPI processes.
\param minimum The minimum permissible number of objects per cell. This value
is typically dictated by maximum objects per cell in data structures that are
used in serial algorithms.
\param targetCellsPerProcess The target number of cells per process. This 
defines the target accuracy for algorithms that use the SFC data structure.

Note that we define this function here so that it is easier to test.
*/
template<std::size_t _Dimension>
std::size_t
maxObjectsPerCellDistributed
(std::size_t const numGlobalObjects,
 int const commSize,
 std::size_t const minimum =
 AdaptiveCells<Traits<_Dimension>, void, false>::DefaultMaxObjectsPerCell,
 std::size_t const targetCellsPerProcess =
 AdaptiveCells<Traits<_Dimension>, void, false>::DefaultTargetCellsPerProcess);


/// Write information about the ordering and the cells.
/**
   \relates AdaptiveCells
*/
template<typename _Traits, typename _Cell, bool _StoreDel>
std::ostream&
operator<<(std::ostream& out,
           AdaptiveCells<_Traits, _Cell, _StoreDel> const& x);


/// Return true if the two sets of cells are compatible.
/**
   \relates AdaptiveCells

   Compatible data structures use the same space-filling curve. Their cells
   overlap only if the cells match. Compatible structures may be merged
   without losing information, that is, without coarsening cells.
*/
template<typename _Traits, typename _Cell, bool _StoreDel>
bool
areCompatible(AdaptiveCells<_Traits, _Cell, _StoreDel> const& a,
              AdaptiveCells<_Traits, _Cell, _StoreDel> const& b);


} // namespace sfc
} // namespace stlib

#define __sfc_AdaptiveCells_tcc__
#include "stlib/sfc/AdaptiveCells.tcc"
#undef __sfc_AdaptiveCells_tcc__

#endif
