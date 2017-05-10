// -*- C++ -*-

#if !defined(__sfc_OrderedCells_tcc__)
#error This file is an implementation detail of OrderedCells.
#endif

namespace stlib
{
namespace sfc
{

//--------------------------------------------------------------------------
// Constructors etc.


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
OrderedCells() :
  _codes(1, Code(_Traits::GuardCode)),
  _cells(1),
  _grid(),
  _objectDelimiters(1, 0)
{
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
OrderedCells(const Point& lowerCorner, const Point& lengths,
             const std::size_t numLevels) :
  _codes(1, Code(_Traits::GuardCode)),
  _cells(1),
  _grid(lowerCorner, lengths, numLevels),
  _objectDelimiters(1, 0)
{
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
OrderedCells(const Grid& order) :
  _codes(1, Code(_Traits::GuardCode)),
  _cells(1),
  _grid(order),
  _objectDelimiters(1, 0)
{
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
OrderedCells(const BBox& tbb, const Float minCellLength) :
  _codes(1, Code(_Traits::GuardCode)),
  _cells(1),
  _grid(tbb, minCellLength),
  _objectDelimiters(1, 0)
{
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
OrderedCells(OrderedCells<_Traits, _Cell, true, _Grid> const& other) :
  _codes(other._codes),
  _cells(other._cells),
  _grid(other._grid),
  _objectDelimiters()
{
  // Copy the object delimiters if necessary.
  _copyObjectDelimiters(other);
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
std::size_t
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
serializedSize() const
{
  return sizeof(std::size_t) + // The number of levels
    sizeof(std::size_t) + // The number of cells.
    sizeof(Code) * _codes.size() + // The codes.
    sizeof(CellRep) * _cells.size() + // The cells.
    sizeof(std::size_t) * _objectDelimiters.size(); // The object delimiters.
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
serialize(std::vector<unsigned char>* buffer) const
{
  // Copy the input state.
  std::vector<unsigned char> const oldBuffer = *buffer;
  // Allocate memory.
  buffer->resize(buffer->size() + serializedSize());
  // Copy in the original buffer.
  memcpy(&(*buffer)[0], &oldBuffer[0],
         oldBuffer.size() * sizeof(unsigned char));
  // Serialize the the cell data in the rest.
  unsigned char* const p = serialize(&(*buffer)[oldBuffer.size()]);
  // Check that the sizes add up.
  if (p != &*buffer->end()) {
    throw std::runtime_error("Serialized data does not match the buffer size "
                             "in OrderedCells::serialize().");
  }
}


// Record the number of levels, the number of cells (excluding the guard
// cell), the codes, and the cells.
template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
unsigned char*
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
serialize(unsigned char* p) const
{
  // The number of levels.
  const std::size_t numLevels_ = numLevels();
  memcpy(p, &numLevels_, sizeof(std::size_t));
  p += sizeof(std::size_t);
  // The number of cells.
  const std::size_t numCells = size();
  memcpy(p, &numCells, sizeof(std::size_t));
  p += sizeof(std::size_t);
  // The codes.
  std::size_t const codesSize = sizeof(Code) * _codes.size();
  memcpy(p, &_codes[0], codesSize);
  p += codesSize;
  // The cells.
  std::size_t const cellsSize = sizeof(CellRep) * _cells.size();
  memcpy(p, &_cells[0], cellsSize);
  p += cellsSize;
  // The object delimiters.
  std::size_t const delimitersSize =
    sizeof(std::size_t) * _objectDelimiters.size();
  memcpy(p, &_objectDelimiters[0], delimitersSize);
  p += delimitersSize;
  return p;
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
unsigned char const*
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
unserialize(unsigned char const* p)
{
  // Read the number of levels, the number of cells (excluding the guard
  // cell), the codes, and the cells.
  // The number of levels.
  std::size_t numLevels_;
  memcpy(&numLevels_, p, sizeof(std::size_t));
  p += sizeof(std::size_t);
  // Set the level in the code functor.
  _grid = Grid(lowerCorner(), lengths(), numLevels_);
  // The number of cells.
  std::size_t numCells;
  memcpy(&numCells, p, sizeof(std::size_t));
  p += sizeof(std::size_t);
  // The codes.
  _codes.resize(numCells + 1);
  const std::size_t codesSize = sizeof(Code) * _codes.size();
  memcpy(&_codes[0], p, codesSize);
  p += codesSize;
  // The cells.
  _cells.resize(numCells + 1);
  const std::size_t cellsSize = sizeof(CellRep) * _cells.size();
  memcpy(&_cells[0], p, cellsSize);
  p += cellsSize;
  // The object delimiters.
  _objectDelimiters.resize(numCells + 1);
  const std::size_t delimitersSize =
    sizeof(std::size_t) * _objectDelimiters.size();
  memcpy(&_objectDelimiters[0], p, delimitersSize);
  p += delimitersSize;
  return p;
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
clear()
{
  _codes.resize(1);
  _codes.back() = _Traits::GuardCode;
  _cells.resize(1);
  _objectDelimiters.resize(1);
  _objectDelimiters.front() = 0;
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
shrink_to_fit()
{
#if (__cplusplus >= 201103L)
  _codes.shrink_to_fit();
  _cells.shrink_to_fit();
#else
  {
    std::vector<Code> copy = _codes;
    _codes.swap(copy);
  }
  {
    CellContainer copy = _cells;
    _cells.swap(copy);
  }
#endif
  _objectDelimiters.shrink_to_fit();
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
checkValidity() const
{
  // There is a code for each cell.
  if (AreStoringCells) {
    assert(_codes.size() == _cells.size());
  }
  checkValidityOfCellCodes(_grid, _codes);
  // Note: The validity of the object delimiters is checked in the derived
  // classes.
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
void
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
calculateObjectCountStatistics(std::size_t* min, std::size_t* max,
                               Float* mean) const
{
  static_assert(_StoreDel, "This function is only available "
                "when storing object delimiters.");
  *min = std::size_t(-1);
  *max = 0;
  *mean = 0;
  for (std::size_t i = 0; i != size(); ++i) {
    std::size_t const count = delimiter(i + 1) - delimiter(i);
    if (count < *min) {
      *min = count;
    }
    if (count > *max) {
      *max = count;
    }
    *mean += count;
  }
  *mean /= size();
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
inline
std::vector<std::size_t>
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
codesToCells(std::vector<Code> const& codes) const
{
#ifdef STLIB_DEBUG
  // The input codes must be in sorted order.
  assert(std::is_sorted(codes.begin(), codes.end(),
                        [](std::size_t a, std::size_t b){ return a <= b; }));
#endif

  std::vector<std::size_t> cells(codes.size());
  // Start with the first cell index.
  std::size_t c = 0;
  // For each of the codes.
  for (std::size_t i = 0; i != codes.size(); ++i) {
    // Advance to the code. Note that there is no need to check
    // for the end of the codes for this data structure because the sequence
    // of codes is a subset.
    for (; _codes[c] != codes[i]; ++c) {
#ifdef STLIB_DEBUG
      // If we reach the guard code, then the input sequence of codes is not
      // a subset of the cells for this class.
      assert(c != size());
#endif
    }
    // Record the cell index.
    cells[i] = c;
  }
  
#ifdef STLIB_DEBUG
  // Check the output.
  for (std::size_t i = 0; i != codes.size(); ++i) {
    assert(_codes[cells[i]] == codes[i]);
  }
#endif
  return cells;
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
template<typename _Object>
inline
void
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
buildCells(std::vector<_Object>* objects, OrderedObjects* orderedObjects)
{
  // Sort the objects.
  std::vector<std::pair<Code, std::size_t> > codeIndexPairs;
  sortByCodes(_grid, objects, &codeIndexPairs);
  // Continue building using the code/index pairs and the sorted objects.
  buildCells(codeIndexPairs, *objects, orderedObjects);
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
template<typename _Object>
inline
void
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
buildCells(std::vector<_Object> const& objects)
{
  // Make a vector of code/index pairs.
  std::vector<std::pair<Code, std::size_t> >
    codeIndexPairs(objects.size());
  // Calculate the codes.
  for (std::size_t i = 0; i != codeIndexPairs.size(); ++i) {
    codeIndexPairs[i].first =
      _grid.code(centroid(geom::specificBBox<BBox>(objects[i])));
    codeIndexPairs[i].second = i;
  }

  // Continue building using the code/index pairs and the sorted objects.
  buildCells(codeIndexPairs, objects);
}


template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Grid>
template<typename _Object>
inline
void
OrderedCells<_Traits, _Cell, _StoreDel, _Grid>::
buildCells(std::vector<std::pair<Code, std::size_t> > const&
           codeIndexPairs,
           std::vector<_Object> const& objects,
           OrderedObjects* orderedObjects)
{
  typedef std::pair<Code, std::size_t> Pair;

  assert(codeIndexPairs.size() == objects.size());
#ifdef STLIB_DEBUG
  assert(std::is_sorted(codeIndexPairs.begin(), codeIndexPairs.end(),
                        [](Pair const& x, Pair const& y)
                        {return x.first < y.first;}));
  // The indices should be a permutation of [0..objects.size()), but we don't 
  // check that. Instead we do a simple sanity check on the individual values.
  for (auto&& pair : codeIndexPairs) {
    assert(pair.second < objects.size());
  }
#endif

  BuildCell<CellRep> buildCell;
  // Count the number of distinct codes so that we know the size for the
  // vectors of codes and cells. Add one for the guard element.
  std::size_t const count =
    ads::countGroups(codeIndexPairs.begin(), codeIndexPairs.end(),
                     [](Pair const& x, Pair const& y)
                     {return x.first == y.first;}) + 1;
  // Build the lists of codes and cells.
  _codes.resize(count);
  _cells.resize(count);
  _objectDelimiters.resize(count);
  std::size_t k = 0;
  for (std::size_t i = 0; i != codeIndexPairs.size(); /*increment inside*/) {
    std::size_t j = i + 1;
    for (; j != codeIndexPairs.size() &&
           codeIndexPairs[j].first == codeIndexPairs[i].first; ++j) {
    }
    _codes[k] = codeIndexPairs[i].first;
    _cells[k] = buildCell(&objects[i], &objects[j]);
    _objectDelimiters[k] = i;
    i = j;
    ++k;
  }
  // Add the guard elements.
  _codes[k] = _Traits::GuardCode;
  _cells[k] = buildCell((const _Object*)(0), (const _Object*)(0));
  _objectDelimiters[k] = codeIndexPairs.size();
  ++k;
  assert(k == _codes.size());

  // Set the ordered indices.
  if (orderedObjects) {
    orderedObjects->set(codeIndexPairs);
  }
}


} // namespace sfc
} // namespace stlib
