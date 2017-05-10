// -*- C++ -*-

#if !defined(__sfc_NonOverlappingCells_tcc__)
#error This file is an implementation detail of NonOverlappingCells.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>::
checkValidity() const
{
  Base::checkValidity();
  checkNonOverlapping(_grid, _codes);
  _checkObjectDelimiters();
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>::
_checkObjectDelimiters() const
{
  if (! _StoreDel) {
    return;
  }
  assert(_codes.size() == _objectDelimiters.size());
  assert(_objectDelimiters.front() == 0);
  // The delimiters must be strictly increasing because empty cells are not
  // stored.
  for (std::size_t i = 1; i != _objectDelimiters.size(); ++i) {
    assert(_objectDelimiters[i] > _objectDelimiters[i - 1]);
  }
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>&
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>::
operator+=(NonOverlappingCells const& other)
{
  assert(Base::numLevels() == other.numLevels());
  NonOverlappingCells copy = *this;
  _merge(copy, other);
  return *this;
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>::
crop(std::vector<std::size_t> const& cellIndices)
{
  // First, update the object delimiters.
  _cropObjectDelimiters(cellIndices, std::integral_constant<bool, _StoreDel>{});

  // Next crop the codes and cells.
  {
    std::vector<Code> codes(cellIndices.size() + 1);
    CellContainer cells(cellIndices.size() + 1);
    for (std::size_t i = 0; i != cellIndices.size(); ++i) {
      codes[i] = _codes[cellIndices[i]];
      cells[i] = _cells[cellIndices[i]];
    }
    // Copy the guard code.
    codes.back() = _codes.back();
    cells.back() = _cells.back();
    _codes.swap(codes);
    _cells.swap(cells);
  }

  // There is no need to alter the ordering class.
}

  
template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>::
_cropObjectDelimiters(std::vector<std::size_t> const& cellIndices,
                      std::true_type /*_StoreDel*/)
{
  ObjectDelimitersContainer objectDelimiters;
  objectDelimiters.reserve(cellIndices.size() + 1);
  objectDelimiters.push_back(0);
  for (auto cell : cellIndices) {
    objectDelimiters.push_back(objectDelimiters.back() +
                               Base::delimiter(cell + 1) -
                               Base::delimiter(cell));
  }
  _objectDelimiters.swap(objectDelimiters);
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
template<typename _Object>
inline
void
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>::
crop(std::vector<std::size_t> const& cells, std::vector<_Object>* objects)
{
  // First update the objects.
  // Count the number of cropped objects.
  std::size_t numCroppedObjects = 0;
  for (auto cell : cells) {
#ifdef STLIB_DEBUG
    assert(cell < Base::size());
#endif
    numCroppedObjects += Base::delimiter(cell + 1) - Base::delimiter(cell);
  }
  // Reserve storage for the cropped objects.
  std::vector<_Object> croppedObjects;
  croppedObjects.reserve(numCroppedObjects);
  // Concatenate the cropped objects.
  for (auto cell : cells) {
    croppedObjects.insert(croppedObjects.end(),
                          &(*objects)[Base::delimiter(cell)],
                          &(*objects)[Base::delimiter(cell + 1)]);
  }
  // Swap to update the objects.
  objects->swap(croppedObjects);

  // Next crop the cells.
  crop(cells);
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>::
_mergeCells()
{
  assert(! _codes.empty());
  if (AreStoringCells) {
    assert(_codes.size() == _cells.size());
  }
  if (_StoreDel) {
    assert(_codes.size() == _objectDelimiters.size());
  }
  // Dispense with the trivial case.
  if (Base::size() == 0) {
    return;
  }

  // The merged cells.
  std::vector<Code> mergedCodes;
  CellContainer mergedCells;
  ObjectDelimitersContainer mergedObjectDelimiters;
  // Initialize the first distinct elements.
  mergedCodes.push_back(_codes[0]);
  Code next = _grid.location(_grid.next(_codes[0]));
  mergedCells.push_back(_cells[0]);
  mergedObjectDelimiters.push_back(_objectDelimiters[0]);
  // Iterate to merge common blocks and add distinct ones.
  for (std::size_t i = 1; i != _codes.size(); ++i) {
    if (_codes[i] < next) {
      mergedCells.back() += _cells[i];
      if (_objectDelimiters[i] < mergedObjectDelimiters.back()) {
        mergedObjectDelimiters.back() = _objectDelimiters[i];
      }
    }
    else {
      mergedCodes.push_back(_codes[i]);
      next = _grid.location(_grid.next(_codes[i]));
      mergedCells.push_back(_cells[i]);
      mergedObjectDelimiters.push_back(_objectDelimiters[i]);
    }
  }
  assert(mergedCodes.back() == _Traits::GuardCode);
  // Swap memory to update the codes, cells, and object delimiters.
  _codes.swap(mergedCodes);
  _cells.swap(mergedCells);
  _objectDelimiters.swap(mergedObjectDelimiters);
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>::
_merge(NonOverlappingCells const& a, NonOverlappingCells const& b)
{
  std::array<NonOverlappingCells const*, 2> inputs = {{&a, &b}};
  // Difference to obtain unlinked information.
  std::array<ObjectDelimitersContainer, 2> inputSizes;
  a._delimitersToSizes(&inputSizes[0]);
  b._delimitersToSizes(&inputSizes[1]);
  // Clear any current cells.
  _codes.clear();
  _cells.clear();
  ObjectDelimitersContainer sizes;
  // Initialize with one cell.
  std::array<std::size_t, 2> i = {{0, 0}};
  std::size_t c = b._codes[0] < a._codes[0];
  if (inputs[c]->size() != 0) {
    _codes.push_back(inputs[c]->_codes[0]);
    _cells.push_back(inputs[c]->_cells[0]);
    sizes.push_back(inputSizes[c][0]);
    ++i[c];
  }
  // Process until we reach both guard elements.
  for (; i[0] != a.size() || i[1] != b.size(); /*increment inside*/) {
    c = b._codes[i[1]] < a._codes[i[0]];
    if (inputs[c]->_codes[i[c]] < _grid.location(_grid.next(_codes.back()))) {
      _cells.back() += inputs[c]->_cells[i[c]];
      sizes.back() += inputSizes[c][i[c]];
    }
    else {
      _codes.push_back(inputs[c]->_codes[i[c]]);
      _cells.push_back(inputs[c]->_cells[i[c]]);
      sizes.push_back(inputSizes[c][i[c]]);
    }
    ++i[c];
  }
  // Add the guard cell.
  _codes.push_back(a._codes.back());
  assert(_codes.back() == _Traits::GuardCode);
  _cells.push_back(a._cells.back());
  _cells.back() += b._cells.back();
  // Convert from cell sizes to object delimiters.
  _sizesToDelimiters(sizes);
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>::
_delimitersToSizes(ObjectDelimitersContainer* sizes) const
{
  if (! _StoreDel) {
    // If we are not storing object delimiters, do nothing.
    return;
  }
  assert(! _objectDelimiters.empty());
  sizes->resize(_objectDelimiters.size() - 1);
  for (std::size_t i = 0; i != sizes->size(); ++i) {
    (*sizes)[i] = _objectDelimiters[i + 1] - _objectDelimiters[i];
  }
}

  
template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
NonOverlappingCells<_Traits, _Cell, _StoreDel, _Grid>::
_sizesToDelimiters(ObjectDelimitersContainer const& sizes)
{
  if (! _StoreDel) {
    // If we are not storing object delimiters, do nothing.
    return;
  }
  _objectDelimiters.resize(sizes.size() + 1);
  _objectDelimiters.front() = 0;
  std::partial_sum(sizes.begin(), sizes.end(), _objectDelimiters.begin() + 1);
}


} // namespace sfc
} // namespace stlib
