// -*- C++ -*-

#if !defined(__sfc_LinearOrthantTrieTopology_tcc__)
#error This file is an implementation detail of LinearOrthantTrieTopology.
#endif

namespace stlib
{
namespace sfc
{


template<std::size_t _Dimension>
inline
LinearOrthantTrieTopology<_Dimension>::
LinearOrthantTrieTopology() :
  _numInternal(0),
  _numLeaves(0),
  _isLeaf(),
  _cellRanks(),
  _children(),
  _parents()
{
}


template<std::size_t _Dimension>
template<typename _Traits, typename _Cell, bool _StoreDel>
inline
LinearOrthantTrieTopology<_Dimension>::
LinearOrthantTrieTopology
(LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie) :
  _numInternal(),
  _numLeaves(),
  _isLeaf(),
  _cellRanks(),
  _children(),
  _parents()
{
  // For each cell record if it is a leaf. Calculate the internal and leaf 
  // ranks.
  auto const sizes =
    calculateBranchAndLeafRanks(trie, &_isLeaf, &_cellRanks);
  _numInternal = sizes.first;
  _numLeaves = sizes.second;
  
  // Cache the children.
  _children.resize(_numInternal);
  for (std::size_t i = 0; i != trie.size(); ++i) {
    if (! _isLeaf[i]) {
      trie.getChildren(i, &_children[_cellRanks[i]]);
    }
  }  

  // Cache the parents.
  calculateParents(trie, &_parents);
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
calculateParents
(LinearOrthantTrie<_Traits, _Cell, _StoreDel>
 const& trie,
 std::vector<std::size_t>* parents)
{
  parents->resize(trie.size());
  if (parents->size() == 0) {
    return;
  }
  // The root cell does not have a parent.
  parents->front() = std::size_t(-1);
  Siblings<_Traits::Dimension> children;
  for (std::size_t i = 0; i != parents->size(); ++i) {
    if (trie.isInternal(i)) {
      trie.getChildren(i, &children);
      for (std::size_t j = 0; j != children.size(); ++j) {
        (*parents)[children[j]] = i;
      }
    }
  }
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
calculateBranchRanks
(LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie,
 std::vector<std::size_t>* ranks)
{
  std::size_t n = 0;
  ranks->resize(trie.size());
  for (std::size_t i = 0; i != trie.size(); ++i) {
    if (trie.isLeaf(i)) {
      (*ranks)[i] = std::size_t(-1);
    }
    else {
      (*ranks)[i] = n++;
    }
  }
  return n;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::size_t
calculateLeafRanks
(LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie,
 std::vector<std::size_t>* ranks)
{
  std::size_t n = 0;
  ranks->resize(trie.size());
  for (std::size_t i = 0; i != trie.size(); ++i) {
    if (trie.isLeaf(i)) {
      (*ranks)[i] = n++;
    }
    else {
      (*ranks)[i] = std::size_t(-1);
    }
  }
  return n;
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::pair<std::size_t, std::size_t>
calculateBranchAndLeafRanks
(LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie,
 std::vector<std::size_t>* branchRanks,
 std::vector<std::size_t>* leafRanks)
{
  std::size_t branch = 0;
  std::size_t leaf = 0;
  branchRanks->resize(trie.size());
  leafRanks->resize(trie.size());
  for (std::size_t i = 0; i != trie.size(); ++i) {
    if (trie.isLeaf(i)) {
      (*branchRanks)[i] = std::size_t(-1);
      (*leafRanks)[i] = leaf++;
    }
    else {
      (*branchRanks)[i] = branch++;
      (*leafRanks)[i] = std::size_t(-1);
    }
  }
  return std::pair<std::size_t, std::size_t>{branch, leaf};
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
std::pair<std::size_t, std::size_t>
calculateBranchAndLeafRanks
(LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie,
 std::vector<bool>* isLeaf,
 std::vector<std::size_t>* ranks)
{
  std::size_t branch = 0;
  std::size_t leaf = 0;
  isLeaf->resize(trie.size());
  ranks->resize(trie.size());
  for (std::size_t i = 0; i != trie.size(); ++i) {
    if (trie.isLeaf(i)) {
      (*isLeaf)[i] = true;
      (*ranks)[i] = leaf++;
    }
    else {
      (*isLeaf)[i] = false;
      (*ranks)[i] = branch++;
    }
  }
  return std::pair<std::size_t, std::size_t>{branch, leaf};
}


} // namespace sfc
} // namespace stlib
