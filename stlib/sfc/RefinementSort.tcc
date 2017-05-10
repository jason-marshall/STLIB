// -*- C++ -*-

#if !defined(__sfc_RefinementSort_tcc__)
#error This file is an implementation detail of RefinementSort.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits>
inline
RefinementSort<_Traits>::
RefinementSort(BlockCode<_Traits> const& blockCode, std::vector<Pair>* pairs,
               std::size_t const maxElementsPerCell) :
  _blockCode(blockCode),
  _pairs(*pairs),
  _maxElementsPerCell(maxElementsPerCell),
  _buffer(pairs->size()),
  _insertIterators()
{
}


template<typename _Traits>
inline
void
RefinementSort<_Traits>::
operator()()
{
  // Because the termination criteria are applied at the end of _sort(),
  // we need to check them before calling at the top level.
  if (_blockCode.numLevels() == 0) {
    return;
  }
  if (_pairs.size() <= _maxElementsPerCell) {
    if (_pairs.size()) {
      _setLevel(0, _pairs.size(), 0);
    }
    return;
  }

  // Sort.
  _sort(0, _pairs.size(), 0);
}


template<typename _Traits>
inline
void
RefinementSort<_Traits>::
_sort(std::size_t begin, std::size_t const end, std::size_t const level)
{
  // The shift for extracting the radix.
  int const shift = Dimension * (_blockCode.numLevels() - level - 1) +
    _blockCode.levelBits();
  // Count the number of elements with each key.
  std::array<std::size_t, Radix> counts;
  std::fill(counts.begin(), counts.end(), 0);
  for (std::size_t i = begin; i != end; ++i) {
    // Extract the radix by left shifting and masking.
    ++counts[(_pairs[i].first >> shift) & Mask];
  }

  // Set iterators for inserting into the buffer.
  _insertIterators[0] = &_buffer[0];
  for (std::size_t i = 1; i != _insertIterators.size(); ++i) {
    _insertIterators[i] = _insertIterators[i - 1] + counts[i - 1];
  }

  // Sort according to the key by using the insert iterators.
  for (std::size_t i = begin; i != end; ++i) {
    *_insertIterators[(_pairs[i].first >> shift) & Mask]++ = _pairs[i];
  }
  // memcpy is faster than std::copy.
  memcpy(&_pairs[begin], &_buffer[0], (end - begin) * sizeof(Pair));

  // When we reach the highest level of refinement, we are done.
  // Recall that the codes had the highest level of refinement on input,
  // so we don't need to set the level.
  if (level + 1 == _blockCode.numLevels()) {
    return;
  }

  // If there are more levels, continue with depth-first recursion.
  for (std::size_t i = 0; i != Radix; ++i) {
    // If the cell has been sufficiently refined.
    if (counts[i] <= _maxElementsPerCell) {
      if (counts[i]) {
        // Set the level.
        _setLevel(begin, begin + counts[i], level + 1);
      }
    }
    else {
      // Otherwise, recurse.
      _sort(begin, begin + counts[i], level + 1);
    }
    begin += counts[i];
  }
}


// Convert to block codes at the specified level.
template<typename _Traits>
inline
void
RefinementSort<_Traits>::
_setLevel(std::size_t const begin, std::size_t const end,
          std::size_t const level)
{
  // The mask erases the location code for higher levels and the level bits.
  Code const mask = Code(-1) <<
  (Dimension * (_blockCode.numLevels() - level) + _blockCode.levelBits());
  // Set the level.
  for (std::size_t i = begin; i != end; ++i) {
    _pairs[i].first = (_pairs[i].first & mask) | level;
  }
}


template<typename _Traits, typename _Object>
inline
void
refinementSort
(BlockCode<_Traits> const& blockCode,
 std::vector<_Object>* objects,
 std::vector<std::pair<typename _Traits::Code, std::size_t> >*
 codeIndexPairs,
 std::size_t const maxElementsPerCell)
{
  typedef typename _Traits::BBox BBox;

  // Allocate memory for the code/index pairs.
  codeIndexPairs->resize(objects->size());
  // Calculate the codes.
  for (std::size_t i = 0; i != codeIndexPairs->size(); ++i) {
    (*codeIndexPairs)[i].first =
      blockCode.code(centroid(geom::specificBBox<BBox>((*objects)[i])));
    (*codeIndexPairs)[i].second = i;
  }
  // Refine and sort by the codes.
  refinementSort(blockCode, codeIndexPairs, maxElementsPerCell);
  // Set the sorted objects.
  {
    std::vector<_Object> obj(codeIndexPairs->size());
    for (std::size_t i = 0; i != codeIndexPairs->size(); ++i) {
      obj[i] = (*objects)[(*codeIndexPairs)[i].second];
    }
    objects->swap(obj);
  }
}


} // namespace sfc
} // namespace stlib
