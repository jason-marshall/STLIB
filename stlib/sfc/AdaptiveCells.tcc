// -*- C++ -*-

#if !defined(__sfc_AdaptiveCells_tcc__)
#error This file is an implementation detail of AdaptiveCells.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits, typename _Cell, bool _StoreDel>
template<bool _Sod>
inline
AdaptiveCells<_Traits, _Cell, _StoreDel>::
AdaptiveCells(UniformCells<_Traits, _Cell, _Sod> const& cellsUniform) :
  Base(cellsUniform.lowerCorner(), cellsUniform.lengths(),
       cellsUniform.numLevels())
{
  // Allocate memory for the codes and cells.
  _codes.resize(cellsUniform.size() + 1);
  _cells.resize(cellsUniform.size() + 1);
  // Convert the location codes to block codes.
  int const levelBits = Base::grid().levelBits();
  Code const numLevels = cellsUniform.numLevels();
  for (std::size_t i = 0; i != _codes.size() - 1; ++i) {
    // Shift the location bits and add in the level.
    _codes[i] = (cellsUniform.code(i) << levelBits) | numLevels;
  }
  // The final element is the guard.
  _codes.back() = _Traits::GuardCode;
  // Simply copy the cells.
  memcpy(&_cells[0], &cellsUniform[0],
         _cells.size() * sizeof(CellRep));
  // Copy the object delimiters.
  Base::_copyObjectDelimiters(cellsUniform);
}


template<typename _Traits, typename _Cell, bool _StoreDel>
template<typename _Object>
inline
void
AdaptiveCells<_Traits, _Cell, _StoreDel>::
buildCells(std::vector<_Object>* objects, std::size_t const maxElementsPerCell,
           OrderedObjects* orderedObjects)
{
  // Refine cells and sort the objects.
  std::vector<std::pair<Code, std::size_t> > codeIndexPairs;
  refinementSort(Base::grid(), objects, &codeIndexPairs, maxElementsPerCell);
  // Continue building using the code/index pairs and the sorted objects.
  Base::buildCells(codeIndexPairs, *objects, orderedObjects);
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
AdaptiveCells<_Traits, _Cell, _StoreDel>::
buildCells(std::vector<std::pair<Code, std::size_t> > const&
           codeSizePairs)
{
  static_assert(! Base::AreStoringCells, "This function may only be used when "
                "you are not storing cells.");
  Base::_codes.resize(codeSizePairs.size());
  for (std::size_t i = 0; i != Base::_codes.size(); ++i) {
    Base::_codes[i] = codeSizePairs[i].first;
  }
  if (_StoreDel) {
    Base::_objectDelimiters.resize(codeSizePairs.size());
    Base::_objectDelimiters.front() = 0;
    for (std::size_t i = 1; i != Base::_objectDelimiters.size(); ++i) {
      Base::_objectDelimiters[i] = Base::_objectDelimiters[i - 1] +
        codeSizePairs[i - 1].second;
    }
  }
#ifdef STLIB_DEBUG
  Base::checkValidity();
#endif
}


template<typename _Traits, typename _Cell, bool _StoreDel>
template<typename _Object>
inline
void
AdaptiveCells<_Traits, _Cell, _StoreDel>::
buildCells(std::vector<Code> const& cellCodes,
           std::vector<Code> const& objectCodes,
           std::vector<_Object> const& objects)
{
  assert(! cellCodes.empty() && cellCodes.back() == _Traits::GuardCode);
  assert(objectCodes.size() == objects.size());
#ifdef STLIB_DEBUG
  assert(std::is_sorted(cellCodes.begin(), cellCodes.end()));
  assert(std::is_sorted(objectCodes.begin(), objectCodes.end()));
#endif

  Base::_codes.clear();
  Base::_cells.clear();
  Base::_objectDelimiters.clear();
  
  BuildCell<CellRep> buildCell;
  // The index of the current cell code.
  std::size_t n = 0;
  // Loop over the objects.
  for (std::size_t i = 0; i != objectCodes.size(); /*increment inside*/) {
    // Scan until we find the cell that contains the first unprocessed object.
    while (cellCodes[n + 1] <= objectCodes[i]) {
      ++n;
    }
    // Scan to determine the range of objects in the cell.
    std::size_t end = i + 1;
    for ( ; end != objectCodes.size() && objectCodes[end] < cellCodes[n + 1];
          ++end) {
    }
    // Build and record the cell.
    Base::_codes.push_back(cellCodes[n]);
    Base::_cells.push_back(buildCell(&objects[i], &objects[end]));
    Base::_objectDelimiters.push_back(i);
    i = end;
  }
  // Record the guard cell.
  Base::_codes.push_back(Code(_Traits::GuardCode));
  Base::_cells.push_back(buildCell(&objects[0], &objects[0]));
  Base::_objectDelimiters.push_back(objects.size());
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
AdaptiveCells<_Traits, _Cell, _StoreDel>::
calculateHighestLevel() const
{
  std::size_t level = 0;
  // Note that we skip the guard code.
  for (std::size_t i = 0; i != Base::size(); ++i) {
    std::size_t const lev = Base::grid().level(_codes[i]);
    if (lev > level) {
      level = lev;
    }
  }
  return level;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
AdaptiveCells<_Traits, _Cell, _StoreDel>::
setNumLevelsToFit()
{
  std::size_t const newLevels = calculateHighestLevel();
  // If we are not going to decrease the number of levels, do nothing.
  if (newLevels == _grid.numLevels()) {
    return;
  }
  // The right shift that will erase the level number and bits for higher
  // levels.
  std::size_t const rightShift = (_grid.numLevels() - newLevels) * Dimension +
    _grid.levelBits();
  // Set the number of levels in the functor for calculating codes.
  _grid.setNumLevels(newLevels);
  // Now we can convert the codes. Right shift to erase the level number
  // and bits for higher levels. Left shift to restore the location, and
  // then add the level number.
  for (auto&& code : _codes) {
    code = (code >> rightShift << _grid.levelBits()) | _grid.level(code);
  }
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
AdaptiveCells<_Traits, _Cell, _StoreDel>::
coarsenWithoutMerging()
{
  // Handle the trivial cases.
  if (Base::size() <= 1) {
    return 0;
  }
  // Note that we don't need to difference/accumulate, because we won't
  // modify the cells. We are just modifying the codes.
  std::size_t count = 0;
  std::size_t c;
  // Apply sweeps of coarsening until no more can be done.
  while ((c = _coarsenWithoutMergingSweep()) != 0) {
    count += c;
  }
  return count;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
template<typename _Predicate>
inline
std::size_t
AdaptiveCells<_Traits, _Cell, _StoreDel>::
coarsen(_Predicate pred)
{
  std::size_t count = 0;
  std::size_t c;
  // Apply sweeps of coarsening until no more groups are coarsened.
  while ((c = _coarsenSweep(pred)) != 0) {
    count += c;
  }
  return count;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
AdaptiveCells<_Traits, _Cell, _StoreDel>::
coarsenMaxCells(std::size_t const maxCells)
{
  assert(maxCells > 0);
  // Shortcut for the trivial case.
  if (Base::size() <= maxCells) {
    return 0;
  }

  // Start by coarsening without merging as it is less expensive than coarsening
  // with a specified maximum cell size.
  std::size_t count = coarsenWithoutMerging();
  // Apply coarsening until the number of cells does not exceed the threshold.
  while (Base::size() > maxCells) {
    count += _coarsenSweepCellSize(_minSizeForCoarsening());
  }
  return count;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
AdaptiveCells<_Traits, _Cell, _StoreDel>::
coarsenCellSize(std::size_t const cellSize)
{
  // First use coarsening without merging because it is a cheap way to
  // start things off when there are many levels of refinement.
  std::size_t count = coarsenWithoutMerging();
  // Apply sweeps of coarsening until no more groups are coarsened.
  // Note that we don't difference because _coarsenSweepCellSize() directly
  // uses cell delimiters.
  std::size_t c;
  while ((c = _coarsenSweepCellSize(cellSize)) != 0) {
    count += c;
  }
  return count;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
bool
AdaptiveCells<_Traits, _Cell, _StoreDel>::
_getSiblings(std::size_t const begin, std::size_t* end, std::size_t* next)
const
{
  // The level for the siblings.
  std::size_t const level = _grid.level(_codes[begin]);
  // The location of the end of the block of siblings.
  Code const endLocation =
    _grid.locationNextParent(_codes[begin]);
  // Try to find a range of siblings to coarsen.
  std::size_t i;
  for (i = begin + 1 ; _grid.level(_codes[i]) == level &&
       _codes[i] < endLocation; ++i) {
  }
  *end = i;

  // If we are now at a lower level, we may need to skip some cells.
  // Find the next position at which we can possibly apply coarsening.
  // We must skip over the block in which we just tried coarsening.
  // Thus, at level l, this location is the next cell at level l - 1.
  while (_grid.level(_codes[i]) < level &&
         _codes[i] <
         _grid.next
         (_grid.atLevel(_codes[begin],
                               _grid.level(_codes[i]) - 1))) {
    ++i;
  }
  *next = i;

  // If we found a range of siblings that may be coarsened.
  return _codes[*end] >= endLocation;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
AdaptiveCells<_Traits, _Cell, _StoreDel>::
_minSizeForCoarsening() const
{
  std::size_t leafSize = -1;
  std::size_t j;
  std::size_t next;
  // For each group of siblings that may be coarsened.
  for (std::size_t i = 0; i != Base::size(); /*increment inside*/) {
    // If there is a range of siblings that may be coarsened.
    if (_getSiblings(i, &j, &next)) {
#ifdef STLIB_DEBUG
      // Running check that the delimiters are valid.
      assert(Base::delimiter(j) > Base::delimiter(i));
#endif
      if (Base::delimiter(j) - Base::delimiter(i) < leafSize) {
        leafSize = Base::delimiter(j) - Base::delimiter(i);
      }
    }
    i = next;
  }
  return leafSize;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
AdaptiveCells<_Traits, _Cell, _StoreDel>::
_coarsenWithoutMergingSweep()
{
  std::size_t count = 0;
  std::size_t j;
  std::size_t next;
  // For each group of siblings that may be coarsened.
  for (std::size_t i = 0; i != Base::size(); /*increment inside*/) {
    // If there is a range of siblings that may be coarsened.
    if (_getSiblings(i, &j, &next)) {
      // If the group is just a single cell.
      if (i + 1 == j) {
        _codes[i] = _grid.parent(_codes[i]);
        ++count;
      }
    }
    i = next;
  }
  return count;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
template<typename _Predicate>
inline
std::size_t
AdaptiveCells<_Traits, _Cell, _StoreDel>::
_coarsenSweep(_Predicate pred)
{
  // Handle the trivial cases.
  if (Base::size() <= 1) {
    return 0;
  }
  std::size_t count = 0;
  std::size_t j;
  std::size_t next;
  // For each group of siblings that may be coarsened.
  for (std::size_t i = 0; i != Base::size(); /*increment inside*/) {
    // If there is a range of siblings that may be coarsened.
    if (_getSiblings(i, &j, &next)) {
      // If the siblings should be coarsened.
      if (pred(_grid.level(_codes[i]), &_cells[i], &_cells[j])) {
        // Change the codes to the parent level so that the cells will be
        // merged later on.
        Code const parent = _grid.parent(_codes[i]);
        for (std::size_t k = i; k != j; ++k) {
          _codes[k] = parent;
        }
        ++count;
      }
    }
    i = next;
  }
  // Merge the coarsened cells if necessary.
  if (count) {
    Base::_mergeCells();
  }
  // Return the number of coarsening operations.
  return count;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
AdaptiveCells<_Traits, _Cell, _StoreDel>::
_coarsenSweepCellSize(std::size_t const cellSize)
{
  // Handle the trivial cases.
  if (Base::size() <= 1) {
    return 0;
  }
  std::size_t count = 0;
  std::size_t j;
  std::size_t next;
  // For each group of siblings that may be coarsened.
  for (std::size_t i = 0; i != Base::size(); /*increment inside*/) {
    // If there is a range of siblings that may be coarsened.
    if (_getSiblings(i, &j, &next)) {
#ifdef STLIB_DEBUG
      assert(Base::delimiter(j) > Base::delimiter(i));
#endif
      // If the siblings should be coarsened.
      if (Base::delimiter(j) - Base::delimiter(i) <= cellSize) {
        // Change the codes to the parent level so that the cells will be
        // merged later on.
        Code const parent = _grid.parent(_codes[i]);
        for (std::size_t k = i; k != j; ++k) {
          _codes[k] = parent;
        }
        ++count;
      }
    }
    i = next;
  }
  // Merge the coarsened cells if necessary.
  if (count) {
    Base::_mergeCells();
  }
  // Return the number of coarsening operations.
  return count;
}


template<typename _AdaptiveCells, typename _Object>
inline
_AdaptiveCells
adaptiveCells(std::vector<_Object>* objects,
                std::size_t const maxObjectsPerCell)
{
  typedef typename _AdaptiveCells::BBox BBox;
  if (objects->empty()) {
    return _AdaptiveCells{};
  }
  // Determine an appropriate domain by putting a bounding box around the
  // objects. Use all available levels of refinement.
  _AdaptiveCells
    result(geom::specificBBox<BBox>(objects->begin(), objects->end()), 0);
  // Build the cells and order the objects.
  result.buildCells(objects, maxObjectsPerCell);
  return result;
}


template<typename _AdaptiveCells, typename _Object>
inline
_AdaptiveCells
adaptiveCells(std::vector<_Object>* objects,
                OrderedObjects* orderedObjects,
                std::size_t const maxObjectsPerCell)
{
  assert(orderedObjects != nullptr);

  typedef typename _AdaptiveCells::BBox BBox;
  if (objects->empty()) {
    orderedObjects->clear();
    return _AdaptiveCells{};
  }
  // Determine an appropriate domain by putting a bounding box around the
  // objects. Use all available levels of refinement.
  _AdaptiveCells
    result(geom::specificBBox<BBox>(objects->begin(), objects->end()), 0);
  // Build the cells and order the objects. Record the original order.
  result.buildCells(objects, maxObjectsPerCell, orderedObjects);
  return result;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::ostream&
operator<<(std::ostream& out,
           AdaptiveCells<_Traits, _Cell, _StoreDel> const& x)
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
    if (AdaptiveCells<_Traits, _Cell, _StoreDel>::
        AreStoringCells) {
      out << ", cell = " << x._cells[i];
    }
    out << '\n';
  }
  return out;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
bool
areCompatible
(AdaptiveCells<_Traits, _Cell, _StoreDel> const& cells0,
 AdaptiveCells<_Traits, _Cell, _StoreDel> const& cells1)
{
  typedef AdaptiveCells<_Traits, _Cell, _StoreDel> Cells;
  typedef typename Cells::Grid Grid;
  typedef typename Cells::Code Code;

  // The structures must use the same space-filling curve.
  if (! (cells0.grid() == cells1.grid())) {
    return false;
  }
  Grid const& order = cells0.grid();

  // Check the trivial case that at least one is empty.
  if (cells0.size() == 0 || cells1.size() == 0) {
    return true;
  }

  // Loop over the two sequences, checking for overlapping blocks that do not
  // match.
  std::size_t i0 = 0;
  std::size_t i1 = 0;
  while (i0 != cells0.size() && i1 != cells1.size()) {
    Code const code0 = cells0.code(i0);
    Code const code1 = cells1.code(i1);
    if (code0 < code1) {
      if (code1 < order.next(code0)) {
        return false;
      }
      ++i0;
    }
    else if (code1 < code0) {
      if (code0 < order.next(code1)) {
        return false;
      }
      ++i1;
    }
    else {
      ++i0;
      ++i1;
    }
  }

  return true;
}


template<std::size_t _Dimension>
inline
std::size_t
maxObjectsPerCellDistributed
(std::size_t const numGlobalObjects,
 int const commSize,
 std::size_t const minimum,
 std::size_t const targetCellsPerProcess)
{
  assert(minimum > 0);
  assert(targetCellsPerProcess > 0);

  // If we assume that we will use the minimum allowed value:
  // The average number of objects per cell.
  std::size_t const averageObjectsPerCellForMinimum =
    std::max(minimum / (1 << _Dimension) * 2, std::size_t(1));
  // Assume that the cells are fairly distributed. This gives us an estimate
  // of the number of cells for a serial algorithm. We use this number of
  // cells as an upper bound on the target number of cells below.
  std::size_t const targetCellsForMinimum =
    numGlobalObjects / averageObjectsPerCellForMinimum;
  // If the target number of cells is fewer than one,
  // just return the minimum acceptable objects per cell.
  if (targetCellsForMinimum == 0) {
    return minimum;
  }
  
  // The target number of cells for a quality grid.
  std::size_t const targetCells =
    std::min(commSize * targetCellsPerProcess, targetCellsForMinimum);
  // The predicted average objects per cell that would result from this choice.
  std::size_t const averageObjectsPerCell =
    std::max(numGlobalObjects / targetCells, std::size_t(1));
  
  return std::max(averageObjectsPerCell * (1 << _Dimension) / 2,
                  minimum);
}


} // namespace sfc
} // namespace stlib
