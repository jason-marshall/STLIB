// -*- C++ -*-

#if !defined(__sfc_LinearOrthantTrieTopology_h__)
#define __sfc_LinearOrthantTrieTopology_h__

/**
  \file
  \brief Cache topological information for a linear orthant trie.
*/

#include "stlib/sfc/LinearOrthantTrie.h"

namespace stlib
{
namespace sfc
{

/// Cache topological information for a linear orthant trie.
/**
   This class is used to make traversal of linear orthant tries more efficient.
   We do this by storing quantities that would otherwise need to be computed
   on demand.
*/
template<std::size_t _Dimension>
class LinearOrthantTrieTopology 
{
  //
  // Constants.
  //
public:

  /// The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;

  //
  // Types.
  //
public:

  /// Siblings are used to store the children of an internal cell.
  typedef stlib::sfc::Siblings<Dimension> Siblings;
  
  //
  // Member data.
  //
private:

  /// The number of internal cells.
  std::size_t _numInternal;
  /// The number of leaves.
  std::size_t _numLeaves;
  /// For each cell, record whether it is a leaf.
  /** \note Storing this information is more efficient than repeatedly calling
      \c LinearOrthantTrie::isLeaf(). */
  std::vector<bool> _isLeaf;
  /// Record the internal and leaf ranks.
  /** This enables one to store vectors of data for the internal cells or for
      the leaves. */
  std::vector<std::size_t> _cellRanks;
  /// Store the cell indices of the children for the internal cells.
  /** This is more efficient than calling
      \c LinearOrthantTrie::getChildren(). */
  std::vector<Siblings> _children;
  /// The parent cell indices.
  /** In LinearOrthantTrie, the only way to find a parent for a
      specified cell is to search backwards in the sequence of
      cells. This is inefficient. Thus, for applications that require
      access to parents, it is best to calculate and store all of
      them. This process has linear computational complexity in the
      number of cells. */
  std::vector<std::size_t> _parents;

public:

  /// The default constructor results in empty containers.
  LinearOrthantTrieTopology();

  /// Construct from a linear orthant trie.
  template<typename _Traits, typename _Cell, bool _StoreDel>
  LinearOrthantTrieTopology
  (LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie);

  /// Return the number of internal cells.
  std::size_t
  numInternal() const
  {
    return _numInternal;
  }

  /// Return the number of leaves.
  std::size_t
  numLeaves() const
  {
    return _numLeaves;
  }

  /// Return true if the specified cell is internal.
  bool
  isInternal(const std::size_t i) const
  {
    return ! _isLeaf[i];
  }

  /// Return true if the specified cell is a leaf.
  bool
  isLeaf(std::size_t const i) const
  {
    return _isLeaf[i];
  }

  /// Return the internal cell rank of the specified cell.
  std::size_t
  internalRank(std::size_t const i) const
  {
#ifdef STLIB_DEBUG
    assert(isInternal(i));
#endif
    return _cellRanks[i];
  }

  /// Return the leaf rank of the specified cell.
  std::size_t
  leafRank(std::size_t const i) const
  {
#ifdef STLIB_DEBUG
    assert(isLeaf(i));
#endif
    return _cellRanks[i];
  }

  /// Return the children of the specified cell.
  /** \pre The cell must be internal. */
  Siblings const&
  children(std::size_t const i) const
  {
    return _children[internalRank(i)];
  }

  /// Return the parent of the specified cell.
  std::size_t
  parent(std::size_t const i) const
  {
    return _parents[i];
  }

  /// Return the required storage for the data structure (in bytes).
  std::size_t
  storage() const
  {
    return 2 * sizeof(std::size_t) +
      _isLeaf.size() / 8 +
      _cellRanks.size() * sizeof(std::size_t) +
      _children.size() * sizeof(Siblings) +
      _parents.size() * sizeof(std::size_t);
  }
};


/// Calculate the parent cell indices.
/** The only way to find a parent for a specified cell is to search backwards
 in the sequence of cells. This is inefficient. Thus, for applications that
 require access to parents, it is best to calculate and store all of them.
 This process has linear computational complexity in the number of cells. */
template<typename _Traits, typename _Cell, bool _StoreDel>
void
calculateParents
(LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie,
 std::vector<std::size_t>* parents);


/// Record the ranks of the branches.
/** If the cell is a branch, the rank is recorded, otherwise the value is
  \c std::size_t(-1). Note that the vector of ranks will be resized. */
template<typename _Traits, typename _Cell, bool _StoreDel>
std::size_t
calculateBranchRanks
(LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie,
 std::vector<std::size_t>* ranks);


/// Record the ranks of the leaves.
/** If the cell is a leaf, the rank is recorded, otherwise the value is
  \c std::size_t(-1). Note that the vector of ranks will be resized. */
template<typename _Traits, typename _Cell, bool _StoreDel>
std::size_t
calculateLeafRanks
(LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie,
 std::vector<std::size_t>* ranks);


/// Record the ranks of the branches and leaves.
template<typename _Traits, typename _Cell, bool _StoreDel>
std::pair<std::size_t, std::size_t>
calculateBranchAndLeafRanks
(LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie,
 std::vector<std::size_t>* branchRanks,
 std::vector<std::size_t>* leafRanks);


/// Record which cells are leaves. Record the ranks of the branches and leaves.
/** If a cell is a leaf, then the leaf rank is recorded. Otherwise the branch
  rank is recorded. */
template<typename _Traits, typename _Cell, bool _StoreDel>
std::pair<std::size_t, std::size_t>
calculateBranchAndLeafRanks
(LinearOrthantTrie<_Traits, _Cell, _StoreDel> const& trie,
 std::vector<bool>* isLeaf,
 std::vector<std::size_t>* ranks);


} // namespace sfc
} // namespace stlib

#define __sfc_LinearOrthantTrieTopology_tcc__
#include "stlib/sfc/LinearOrthantTrieTopology.tcc"
#undef __sfc_LinearOrthantTrieTopology_tcc__

#endif
