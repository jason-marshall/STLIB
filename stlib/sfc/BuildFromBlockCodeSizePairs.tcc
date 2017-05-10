// -*- C++ -*-

#if !defined(__sfc_BuildFromBlockCodeSizePairs_tcc__)
#error This file is an implementation detail of BuildFromBlockCodeSizePairs.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits>
inline
BuildFromBlockCodeSizePairs<_Traits>::
BuildFromBlockCodeSizePairs(BlockCode<_Traits> const& blockCode,
                            std::vector<Pair> const& input,
                            std::size_t const maxElementsPerCell,
                            std::vector<Pair>* const output) :
  _blockCode(blockCode),
  _input(input),
  _maxElementsPerCell(maxElementsPerCell),
  _output(output)
{
#ifdef STLIB_DEBUG
  // Check the validity of the input codes.
  checkValidityOfCellCodes(_blockCode, _input);
#endif
  // The guard element does not hold any objects.
  assert(_input.back().second == 0);
}


template<typename _Traits>
inline
void
BuildFromBlockCodeSizePairs<_Traits>::
operator()() const
{
  _output->clear();
  // Start with the top-level code and the first code/size pair.
  _build(0, 0);
  // Add the guard code.
  _output->push_back(Pair{_input.back().first, 0});
#ifdef STLIB_DEBUG
  checkValidityOfCellCodes(_blockCode, *_output);
#endif
}


template<typename _Traits>
inline
std::size_t
BuildFromBlockCodeSizePairs<_Traits>::
_build(Code const code, std::size_t i) const
{
  // The location of the next cell.
  Code const next = _blockCode.location(_blockCode.next(code));
  // If there is nothing in this cell, return.
  if (_input[i].first >= next) {
    return i;
  }
  // If we reach an exact match, then we add the cell, regardless of the object
  // count.
  if (code == _input[i].first) {
    _output->push_back(_input[i]);
    return i + 1;
  }
  // Scan forward, counting the objects in this cell.
  std::size_t j = i + 1;
  std::size_t count = _input[i].second;
  for ( ; _input[j].first < next &&
         (count += _input[j].second) <= _maxElementsPerCell; ++j) {
  }
  // If there is a single cell in this block, or if the number of objects
  // does not exceed the threshold, add this block code.
  if (_input[j].first >= next) {
    _output->push_back(Pair{code, count});
    return j;
  }
  // Recurse for each of the children.
  assert(_blockCode.level(code) != _blockCode.numLevels());
  for (Code child = code + 1; child < next;
       child = _blockCode.next(child)) {
    i = _build(child, i);
  }
  return i;
}


} // namespace sfc
} // namespace stlib
