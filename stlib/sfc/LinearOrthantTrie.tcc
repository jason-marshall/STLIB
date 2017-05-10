// -*- C++ -*-

#if !defined(__sfc_LinearOrthantTrie_tcc__)
#error This file is an implementation detail of LinearOrthantTrie.
#endif

namespace stlib
{
namespace sfc
{


template<typename _Traits, typename _Cell, bool _StoreDel>
template<bool _Sod>
inline
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
LinearOrthantTrie(AdaptiveCells<_Traits, _Cell, _Sod> const&
                  adaptiveCells) :
  // Copy the codes and the cells.
  Base(adaptiveCells),
  _next()
{
  // Insert the internal nodes and link to next cells.
  _insertInternal();
}


template<typename _Traits, typename _Cell, bool _StoreDel>
template<bool _Sod>
inline
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
LinearOrthantTrie(UniformCells<_Traits, _Cell, _Sod> const& cellsUniform) :
  // Convert the codes. Then copy the codes and the cells.
  Base(AdaptiveCells<_Traits, _Cell, _Sod>(cellsUniform)),
  _next()
{
  // Insert the internal nodes and link to next cells.
  _insertInternal();
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
countLeaves() const
{
  std::size_t numLeaves = 0;
  for (std::size_t i = 0; i != Base::size(); ++i) {
    if (isLeaf(i)) {
      ++numLeaves;
    }
  }
  return numLeaves;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
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
std::size_t
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
calculateMaxLeafSize() const
{
  std::size_t size = 0;
  for (std::size_t i = 0; i != Base::size(); ++i) {
    if (isLeaf(i)) {
      // Note that this formula only works for leaves.
      std::size_t const sz = Base::delimiter(i + 1) - Base::delimiter(i);
      if (sz > size) {
        size = sz;
      }
    }
  }
  return size;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
template<typename _Object>
inline
void
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
buildCells(std::vector<_Object>* objects, std::size_t const maxElementsPerCell,
           OrderedObjects* orderedObjects)
{
  // Refine cells and sort the objects.
  std::vector<std::pair<Code, std::size_t> > codeIndexPairs;
  refinementSort(Base::grid(), objects, &codeIndexPairs,
                 maxElementsPerCell);
  // Continue building using the code/index pairs and the sorted objects.
  Base::buildCells(codeIndexPairs, *objects, orderedObjects);
  _insertInternal();
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
clear()
{
  Base::clear();
  _next.resize(1);
  _next.back() = std::size_t(-1);
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
shrink_to_fit()
{
  Base::shrink_to_fit();
#if (__cplusplus >= 201103L)
  _next.shrink_to_fit();
#else
  {
    std::vector<std::size_t> copy = _next;
    _next.swap(copy);
  }
#endif
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
checkValidity() const
{
  // First check the validity of the cells.
  Base::checkValidity();
  _checkObjectDelimiters();
  // Then check the trie structure.
  // The next cells must not be at a higher level.
  for (std::size_t i = 0; i != Base::size(); ++i) {
    std::size_t const next = _next[i];
    assert(next > i);
    if (next != Base::size()) {
      assert(_grid.level(_codes[i]) >=
             _grid.level(_codes[next]));
    }
  }
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
_checkObjectDelimiters() const
{
  if (! _StoreDel) {
    return;
  }
  assert(_codes.size() == _objectDelimiters.size());
  assert(_objectDelimiters.front() == 0);
  assert(is_sorted(_objectDelimiters.begin(), _objectDelimiters.end()));
  // Each cell must be non-empty.
  for (std::size_t i = 0; i != Base::size(); ++i) {
    assert(_objectDelimiters[next(i)] > _objectDelimiters[i]);
  }
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
_insertInternal()
{
  // Check the trivial case that the data structure is uninitialized.
  if (Base::empty()) {
    return;
  }

  // The set of new parents that we will add.
  std::unordered_set<Code> parents;
  // The cells for which we need to add parents.
  // Initialize the to-do list with the leaves.
  std::vector<Code> todo(_codes);
  // Get rid of the guard code.
  todo.pop_back();

  // Process the to-do list.
  while (! todo.empty()) {
    const Code code = todo.back();
    todo.pop_back();
    if (_grid.level(code) != 0) {
      const Code parentCode = _grid.parent(code);
      // If this is the first time we have encountered this cell.
      if (parents.insert(parentCode).second) {
        // We will have to add its parents.
        todo.push_back(parentCode);
      }
    }
  }

  // Insert the parent cells. Note that the parent cells do not yet hold
  // valid information. Only the codes are correct.
  _codes.insert(_codes.end(), parents.begin(), parents.end());
  _cells.insert(_cells.end(), parents.size(), _cells.back());
  _objectDelimiters.insert(_objectDelimiters.end(), parents.size(),
                                 std::size_t(-1));
  // Sort the cells to put the parents in the correct positions.
  _sortCells();
  // Add links to the next cells at each level.
  _linkNext();
  // Merge information from children into parents.
  _mergeToParents();
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
_sortCells()
{
  if (Base::AreStoringCells) {
    assert(_codes.size() == _cells.size());
  }
  if (_StoreDel) {
    assert(_codes.size() == _objectDelimiters.size());
  }
  // Make a vector of code/index pairs.
  std::vector<std::pair<Code, std::size_t> > pairs(_codes.size());
  for (std::size_t i = 0; i != pairs.size(); ++i) {
    pairs[i].first = _codes[i];
    pairs[i].second = i;
  }
  // Sort by the codes.
  lorg::sort(&pairs, _grid.numBits());

  // Set the sorted codes, cells, and object delimiters.
  typename Base::CellContainer c(pairs.size());
  typename Base::ObjectDelimitersContainer od(pairs.size());
  for (std::size_t i = 0; i != pairs.size(); ++i) {
    _codes[i] = pairs[i].first;
    c[i] = _cells[pairs[i].second];
    od[i] = _objectDelimiters[pairs[i].second];
  }
  _cells.swap(c);
  _objectDelimiters.swap(od);
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
_eraseInternal()
{
  std::vector<Code> leafCodes;
  typename Base::CellContainer leafCells;
  // Get the leaf cells.
  for (std::size_t i = 0; i != Base::size(); ++i) {
    if (isLeaf(i)) {
      leafCodes.push_back(_codes[i]);
      leafCells.push_back(_cells[i]);
    }
  }
  // Add the guard cell.
  leafCodes.push_back(_codes.back());
  leafCells.push_back(_cells.back());
  // Swap to get the leaves.
  _codes.swap(leafCodes);
  _cells.swap(leafCells);
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
_linkNext()
{
  // Resize to match the sequence of cells, including the guard.
  _next.resize(_codes.size());
  // Loop over the cells. (Not including the guard cell.) Maintain a stack 
  // of the previous cells that do not yet have a next link. The n_th cell
  // in the previous stack is at level n.
  std::vector<std::size_t> previous;
  for (std::size_t i = 0; i != Base::size(); ++i) {
    std::size_t const level = _grid.level(_codes[i]);
    // Process previous cells from this level up.
    while (previous.size() > level) {
      _next[previous.back()] = i;
      previous.pop_back();
    }
    // Add this cell to the previous stack.
    previous.push_back(i);
  }  
  // For cells without a next cell, link to the guard cell.
  for (std::size_t i = 0; i != previous.size(); ++i) {
    _next[previous[i]] = Base::size();
  }
  // Invalidate the next link for the guard cell.
  _next.back() = std::size_t(-1);
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
LinearOrthantTrie<_Traits, _Cell, _StoreDel>::
_mergeToParents()
{
  // Record the cells at each level.
  std::vector<std::vector<std::size_t> > cellsAtLevel(Base::numLevels() + 1);
  for (std::size_t i = 0; i != Base::size(); ++i) {
    std::size_t const level = _grid.level(_codes[i]);
    cellsAtLevel[level].push_back(i);
  }

  Siblings<Dimension> children;
  // For each level of parents. Iterate from highest to lowest.
  for (std::size_t level = Base::numLevels() - 1; level != std::size_t(-1);
       --level) {
    // Iterate through the cells at this level.
    for (std::size_t i = 0; i != cellsAtLevel[level].size(); ++i) {
      std::size_t const parent = cellsAtLevel[level][i];
      // If this is an internal cell.
      if (isInternal(parent)) {
        // Set the cell information by merging the children.
        getChildren(parent, &children);
        _cells[parent] = cellMerge(_cells, children.begin(), children.end());
        // The first child immediately follows the parent. They share the
        // same object delimiter value.
        _objectDelimiters[parent] = _objectDelimiters[parent + 1];
      }
    }
  }
}


} // namespace sfc
} // namespace stlib
