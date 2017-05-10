// -*- C++ -*-

#if !defined(__sfc_RefinementSortCodes_tcc__)
#error This file is an implementation detail of RefinementSortCodes.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits>
inline
RefinementSortCodes<_Traits>::
RefinementSortCodes(BlockCode<_Traits> const& blockCode,
                    std::vector<Code>* codes,
                    std::size_t const maxElementsPerCell) :
  _blockCode(blockCode),
  _codes(*codes),
  _maxElementsPerCell(maxElementsPerCell),
  _buffer(codes->size()),
  _insertIterators(),
  _highestLevel()
{
#ifdef STLIB_DEBUG
  // Check the validity of the input codes.
  for (auto code: _codes) {
    assert(_blockCode.isValid(code));
    // The input codes must be at the highest level of refinement.
    assert(_blockCode.level(code) == _blockCode.numLevels());
  }
#endif
}


template<typename _Traits>
inline
std::size_t
RefinementSortCodes<_Traits>::
operator()()
{
  _highestLevel = 0;
  // Because the termination criteria are applied at the end of _sort(),
  // we need to check them before calling it at the top level.
  if (_blockCode.numLevels() > 0 && _codes.size() > _maxElementsPerCell) {
    _sort(0, _codes.size(), 0);
  }
  return _highestLevel;
}


template<typename _Traits>
inline
void
RefinementSortCodes<_Traits>::
_sort(std::size_t begin, std::size_t const end, std::size_t level)
{
  // The shift for extracting the radix.
  int const shift = Dimension * (_blockCode.numLevels() - level - 1) +
    _blockCode.levelBits();
  // Count the number of elements with each key.
  std::array<std::size_t, Radix> counts;
  std::fill(counts.begin(), counts.end(), 0);
  for (std::size_t i = begin; i != end; ++i) {
    // Extract the radix by left shifting and masking.
    ++counts[(_codes[i] >> shift) & Mask];
  }

  // Set iterators for inserting into the buffer.
  _insertIterators[0] = &_buffer[0];
  for (std::size_t i = 1; i != _insertIterators.size(); ++i) {
    _insertIterators[i] = _insertIterators[i - 1] + counts[i - 1];
  }

  // Sort according to the key by using the insert iterators.
  for (std::size_t i = begin; i != end; ++i) {
    *_insertIterators[(_codes[i] >> shift) & Mask]++ = _codes[i];
  }
  // memcpy is faster than std::copy.
  memcpy(&_codes[begin], &_buffer[0], (end - begin) * sizeof(Code));

  // The children are one level higher.
  ++level;
  if (level > _highestLevel) {
    _highestLevel = level;
  }

  // When we reach the highest level of refinement, we are done.
  // Recall that the codes had the highest level of refinement on input,
  // so we don't need to set the level.
  if (level == _blockCode.numLevels()) {
    return;
  }

  // If there are more levels, continue with depth-first recursion.
  for (std::size_t i = 0; i != Radix; ++i) {
    // If the cell has not been sufficiently refined.
    if (counts[i] > _maxElementsPerCell) {
      // Recurse.
      _sort(begin, begin + counts[i], level);
    }
    begin += counts[i];
  }
}


template<typename _Traits>
inline
void
objectCodesToCellCodeCountPairs
(std::vector<typename _Traits::Code> const& objectCodes,
 std::vector<std::pair<typename _Traits::Code, std::size_t> >*
 codeCountPairs)
{
  typedef typename _Traits::Code Code;
  typedef std::pair<Code, std::size_t> Pair;
  Code const GuardCode = _Traits::GuardCode;

  codeCountPairs->clear();
  // Handle the trivial case.
  if (objectCodes.empty()) {
    codeCountPairs->push_back(Pair{GuardCode, 0});
    return;
  }

  // Start with a pair with zero count to simplify the following logic in 
  // the loop.
  codeCountPairs->push_back(Pair{objectCodes[0], 0});
  // Convert object codes to code/count pairs.
  for (auto x : objectCodes) {
    if (codeCountPairs->back().first == x) {
      ++codeCountPairs->back().second;
    }
    else {
      codeCountPairs->push_back(Pair{x, 1});
    }
  }
  // Sequences of cell codes are terminated with the guard code.
  codeCountPairs->push_back(Pair{GuardCode, 0});
}


} // namespace sfc
} // namespace stlib
