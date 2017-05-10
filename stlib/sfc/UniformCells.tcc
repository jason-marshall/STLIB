// -*- C++ -*-

#if !defined(__sfc_UniformCells_tcc__)
#error This file is an implementation detail of UniformCells.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
UniformCells<_Traits, _Cell, _StoreDel>::
setNumLevels(const std::size_t levels)
{
  // Short-cut for the trivial case.
  if (levels == Base::numLevels()) {
    return;
  }

  assert(levels <= Base::numLevels());
  // Shift the location codes.
  const int shift = (Base::numLevels() - levels) * Dimension;
  for (std::size_t i = 0; i != Base::size(); ++i) {
    _codes[i] >>= shift;
  }
  // Reduce the level by one in the code functor.
  _grid.setNumLevels(levels);
  // Merge cells with the same code.
  Base::_mergeCells();
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
UniformCells<_Traits, _Cell, _StoreDel>::
coarsen()
{
  assert(Base::numLevels() != 0);
  setNumLevels(Base::numLevels() - 1);
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
UniformCells<_Traits, _Cell, _StoreDel>::
coarsen(const std::size_t cellSize)
{
  std::size_t count = 0;
  while (_shouldCoarsen(cellSize)) {
    coarsen();
    ++count;
  }
  return count;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
bool
UniformCells<_Traits, _Cell, _StoreDel>::
_shouldCoarsen(std::size_t const cellSize) const
{
  static_assert(_StoreDel, "You must store the object delimiters "
                "to use _shouldCoarsen().");
  // Check the trivial case.
  if (Base::numLevels() == 0) {
    return false;
  }
  // For each group of siblings that may be coarsened.
  for (std::size_t i = 0; i != Base::size(); /*increment inside*/) {
    Code const nextParent = _grid.nextParent(_codes[i]);
    std::size_t const begin = Base::delimiter(i);
    for (; _codes[i] < nextParent; ++i) {
    }
    if (Base::delimiter(i) - begin > cellSize) {
      return false;
    }
  }
  return true;
}


} // namespace sfc
} // namespace stlib
