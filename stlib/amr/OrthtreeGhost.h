// -*- C++ -*-

/*!
  \file amr/Orthtree.h
  \brief An orthant tree that uses std::map .
*/

#if !defined(__amr_Orthtree_h__)
#define __amr_Orthtree_h__

#include "stlib/ads/functor/constant.h"
#include "stlib/ads/iterator/TrivialOutputIterator.h"
#include "stlib/numerical/partition.h"
#include "stlib/container/MultiArray.h"

#include <map>
#include <set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace stlib
{
namespace amr
{

//! An orthant tree that uses std::map .
/*!
  \param _Patch The data held in each node.
  \param _Traits The traits class defines the space dimension, the maximum
  level, the spatial index, and the real number type.

  In N-D space, an \e orthant is a region of space obtained by constraining each
  coordinate to be either positive or negative.  Thus there are 2<sup>N</sup>
  orthants.  In 2-D and 3-D one typically uses the more familiar terms
  quadrant and octant.

  A quadtree divides space into quadrants; an octree divides space into
  octants.  Since this data structure divides N-D space into orthants,
  it must be an \e orthtree.

  If supported, some of the patches in the tree may be ghost patches. No
  operations modify ghost patches.
*/
template<typename _Patch, class _Traits>
class Orthtree
{
  //
  // Private types.
  //
private:

  typedef std::map<typename _Traits::SpatialIndex, _Patch> Map;

  //
  // Public types and enumerations.
  //
public:

  //! The element type.
  typedef typename Map::mapped_type Patch;
  //! The orthtree traits.
  typedef _Traits Traits;
  //! The space dimension and the maximum level.
  enum {Dimension = Traits::Dimension, MaximumLevel = Traits::MaximumLevel,
        NumberOfOrthants = Traits::NumberOfOrthants
       };
  //! The spatial index is the key type for the map data structure.
  typedef typename Traits::SpatialIndex SpatialIndex;
  //! The number type.
  typedef typename Traits::Number Number;
  //! A Cartesian point.
  typedef typename Traits::Point Point;

  //! The value type for the tree.
  typedef typename Map::value_type value_type;
  //! An iterator in the tree.
  typedef typename Map::iterator iterator;
  //! A const iterator in the tree.
  typedef typename Map::const_iterator const_iterator;
  //! The size type.
  typedef typename Map::size_type size_type;

  //
  // Protected types.
  //
protected:

  //! An integer type that can hold the level.
  typedef typename SpatialIndex::Level Level;
  //! An integer type that can hold a binary coordinate.
  typedef typename SpatialIndex::Coordinate Coordinate;
  //! An integer type that can hold the interleaved coordinate code.
  typedef typename SpatialIndex::Code Code;

  //
  // Member data.
  //
private:

  //! The nodes in the tree.
  Map _nodes;
  //! The Cartesian lower corner.
  Point _lowerCorner;
  //! The Cartesian extents of a node at each level.
  std::array < Point, MaximumLevel + 1 > _extents;

  //
  // Nested classes.
  //
public:

  //! Compare iterators by their codes.
  struct CompareIterator {
    bool
    operator()(const const_iterator i, const const_iterator j) const
    {
      return i->first < j->first;
    }
  };

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Make an empty tree.
  Orthtree(const Point& lowerCorner, const Point& extents);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the beginning of the range of nodes.
  const_iterator
  begin() const
  {
    return _nodes.begin();
  }

  //! Return the end of the range of nodes.
  const_iterator
  end() const
  {
    return _nodes.end();
  }

  //! Return the maximum possible size.
  size_type
  max_size() const
  {
    return _nodes.max_size();
  }

  //! Get the number of non-ghost nodes.
  /*!
    If ghost nodes are supported, count the number of non-ghost nodes.
    Otherwise call the size() function from the base class.
  */
  size_type
  size() const
  {
    return _size(std::integral_constant<bool, Traits::HasGhost>());
  }

  //! Return true if the tree is empty.
  /*!
    If ghost nodes are supported, return true if there are no non-ghost nodes.
    Otherwise call the empty() function from the base class.
  */
  bool
  empty() const
  {
    return _empty(std::integral_constant<bool, Traits::HasGhost>());
  }

  //! Return true if it is a ghost node.
  bool
  isGhost(const const_iterator node) const
  {
    return _isGhost(node, std::integral_constant<bool, Traits::HasGhost>());
  }

  //! Get the lower corner of the Cartesian domain of the tree.
  const Point&
  getLowerCorner() const
  {
    return _lowerCorner;
  }

  //! Get the extents of the Cartesian domain of the tree.
  const Point&
  getExtents() const
  {
    return _extents[0];
  }

  //! Compute the lower corner of the leaf.
  void
  computeLowerCorner(const SpatialIndex& key, Point* lowerCorner) const;

  //! Compute the lower corner of the leaf.
  Point
  computeLowerCorner(const SpatialIndex& key) const
  {
    Point lowerCorner;
    computeLowerCorner(key, &lowerCorner);
    return lowerCorner;
  }

  //! Get the Cartesian extents of the leaf.
  const Point&
  getExtents(const SpatialIndex& key) const
  {
    return _extents[key.getLevel()];
  }

  // CONTINUE: Rename.
  //! Return true if the element can be refined.
  bool
  canBeRefined(const const_iterator element) const
  {
    return element->first.canBeRefined();
  }

  //! Get the keys that are parents of 2<sup>Dimension</sup> leaves.
  template<typename _OutputIterator>
  void
  getParentKeys(_OutputIterator parentKeys) const;

  //! Get the keys that are parents of 2<sup>Dimension</sup> leaves and would result in a balanced tree under merging.
  template<typename _OutputIterator>
  void
  getParentKeysBalanced(_OutputIterator parentKeys) const;

  //! Return true if the tree is balanced.
  bool
  isBalanced() const;

  //! Get the adjacent neighbors in the specified direction in a balanced tree.
  /*!
    Since the tree is balanced, the level of the adjacent neighbor(s) is
    within one level of the specified node.

    Write a \c const_iterator to each adjacent neighbor to the output iterator.
  */
  template<typename _OutputIterator>
  void
  getBalancedNeighbors(const const_iterator node, _OutputIterator output)
  const;

  //! Get the adjacent neighbors in the specified direction in a balanced tree.
  /*!
    Since the tree is balanced, the level of the adjacent neighbor(s) is
    within one level of the specified node.

    Write a \c const_iterator to each adjacent neighbor to the output iterator.
  */
  template<typename _OutputIterator>
  void
  getBalancedNeighbors(const const_iterator node,
                       const std::size_t neighborDirection,
                       _OutputIterator output) const;

private:

  //! Get the number of nodes. There are no ghost nodes.
  size_type
  _size(std::false_type /*HasGhost*/) const
  {
    return _nodes.size();
  }

  //! Get the number of non-ghost nodes.
  size_type
  _size(std::true_type /*HasGhost*/) const
  {
    size_type size = 0;
    for (const_iterator i = begin(); i != end(); ++i) {
      size += ! isGhost(i);
    }
    return size;
  }

  //! Return true if the tree is empty.
  bool
  _empty(std::false_type /*HasGhost*/) const
  {
    return _nodes.empty();
  }

  //! Return true if there are no non-ghost nodes.
  bool
  _empty(std::true_type /*HasGhost*/) const
  {
    // Look for a non-ghost node.
    for (const_iterator i = begin(); i != end(); ++i) {
      if (! isGhost(i)) {
        return false;
      }
    }
    // If we did not find one, the tree is empty.
    return true;
  }

  //! Return false because there are no ghost nodes.
  bool
  _isGhost(const const_iterator node, std::false_type /*HasGhost*/)
  const
  {
    return false;
  }

  //! Return true if it is a ghost node.
  bool
  _isGhost(const const_iterator node, std::true_type /*HasGhost*/)
  const
  {
#ifdef STLIB_DEBUG
    assert(node != end());
#endif
    return node->second.isGhost();
  }

  //! Advance the iterator. Ignore ghost nodes.
  template<typename _Distance>
  void
  advance(iterator* node, _Distance n)
  {
    _advance(node, n, std::integral_constant<bool, Traits::HasGhost>());
  }

  //! Advance the iterator.
  template<typename _Distance>
  void
  _advance(iterator* node, _Distance n, std::false_type /*HasGhost*/)
  {
    std::advance(*node, n);
  }

  //! Advance the iterator. Ignore ghost nodes.
  template<typename _Distance>
  void
  _advance(iterator* node, _Distance n, std::true_type /*HasGhost*/)
  {
    if (n > 0) {
      while (n) {
        ++*node;
        if (! isGhost(*node)) {
          --n;
        }
      }
    }
    else if (n < 0) {
      while (n) {
        --*node;
        if (! isGhost(*node)) {
          ++n;
        }
      }
    }
  }

  //! Return true if a node with the given key needs refinement in order to balance the tree.
  /*!
    A node needs refinement if it has an adjacent neighbor that is more than
    one level higher than itself.
  */
  bool
  needsRefinementToBalance(const SpatialIndex& key) const;

  //! Return true if the node has a higher level neighbor in the specified direction.
  bool
  hasHigherNeighbor(const const_iterator node, const std::size_t direction)
  const;

  //! Return true if the node has a higher level neighbor.
  bool
  hasHigherNeighbor(const const_iterator node) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name Equality.
  //@{
public:

  //! Return true if the data structures are equal.
  bool
  operator==(const Orthtree& other)
  {
    return _nodes == other._nodes &&
           _lowerCorner == other._lowerCorner &&
           _extents == other._extents;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Return the beginning of the range of nodes.
  iterator
  begin()
  {
    return _nodes.begin();
  }

  //! Return the end of the range of nodes.
  iterator
  end()
  {
    return _nodes.end();
  }

  //! Erase all of the elements.
  void
  clear()
  {
    _nodes.clear();
  }

  //! Set whether this is a ghost patch.
  /*!
    \pre Traits::HasGhost must be true and the patch must have a setGhost()
    function. Otherwise using this function will cause a compilation error.
  */
  void
  setGhost(const iterator node, const bool isGhost = true)
  {
    node->second.setGhost(isGhost);
  }

  //! Insert a value type (key-element pair).
  iterator
  insert(const value_type& x);

  //! Insert a node with the specified spatial index.
  iterator
  insert(const SpatialIndex& spatialIndex)
  {
    return insert(value_type(spatialIndex, Patch()));
  }

  //! Insert a node with the specified spatial index and patch.
  iterator
  insert(const SpatialIndex& key, const Patch& patch)
  {
    return insert(value_type(key, patch));
  }

  //! Given a hint to the position, insert a node with the specified spatial index and patch.
  iterator
  insert(const iterator position, const SpatialIndex& key, const Patch& patch)
  {
    return _nodes.insert(position, value_type(key, patch));
  }

  //! Insert a range of values.
  template<typename InputIterator>
  void
  insert(InputIterator begin, InputIterator end)
  {
    while (begin != end) {
      insert(*begin++);
    }
  }

  //! Erase a value.
  void
  erase(iterator node)
  {
    _nodes.erase(node);
  }

  //! Erase an element according to the provided key.
  size_type
  erase(const SpatialIndex& key)
  {
    return _nodes.erase(key);
  }

  //! Erase a range of elements.
  void
  erase(iterator first, iterator last)
  {
    _nodes.erase(first, last);
  }

  //! Split a node. Return the first child.
  iterator
  split(const iterator parent)
  {
    return split(parent, ads::constructTrivialOutputIterator());
  }

  //! Split a node.  Get the children. Return the first child.
  template<typename _OutputIterator>
  iterator
  split(iterator parent, _OutputIterator children);

  //! Merge the nodes given the first child.  Return the merged node.
  /*!
    \pre All of the children must be present.
  */
  iterator
  merge(iterator firstChild);

  //! Perform refinement to balance the tree.
  /*!
    \return The number of refinement operations.
  */
  std::size_t
  balance()
  {
    return balance(ads::constructTrivialOutputIterator());
  }

  //! Perform refinement to balance the tree. Record the new nodes.
  /*!
    \return The number of refinement operations.
  */
  template<typename _OutputIterator>
  std::size_t
  balance(_OutputIterator newNodes);

  //! Get the non-ghost, adjacent neighbors which have lower levels.
  template<typename _OutputIterator>
  void
  getLowerNeighbors(const iterator node, _OutputIterator i);

  //! Get the mergeable groups of 2<sup>Dimension</sup> nodes.
  template<typename _OutputIterator>
  void
  getMergeableGroups(_OutputIterator lowerCornerNodes)
  {
    getMergeableGroups(lowerCornerNodes, begin(), end());
  }

  //! Get the mergeable groups of 2<sup>Dimension</sup> nodes from the range of nodes.
  template<typename _OutputIterator>
  void
  getMergeableGroups(_OutputIterator lowerCornerNodes, iterator start,
                     iterator finish);

  //! Get the mergeable groups of 2<sup>Dimension</sup> nodes whose merging would result in a balanced tree.
  template<typename _OutputIterator>
  void
  getMergeableGroupsBalanced(_OutputIterator lowerCornerNodes)
  {
    getMergeableGroupsBalanced(lowerCornerNodes, begin(), end());
  }

  //! Get the mergeable groups of 2<sup>Dimension</sup> nodes from the range of nodes whose merging would result in a balanced tree.
  template<typename _OutputIterator>
  void
  getMergeableGroupsBalanced(_OutputIterator lowerCornerNodes, iterator start,
                             iterator finish);

  //@}
  //--------------------------------------------------------------------------
  //! \name Search.
  //@{
public:

  //! Find a value.
  const_iterator
  find(const SpatialIndex& key) const
  {
    return _nodes.find(key);
  }

  //! Find a value.
  iterator
  find(const SpatialIndex& key)
  {
    return _nodes.find(key);
  }

  //! Find the node that matches the code.  If the node is not in the tree, find its ancestor.
  const_iterator
  findAncestor(const SpatialIndex& key) const;

  //! Find the node that matches the code.  If the node is not in the tree, find its ancestor.
  iterator
  findAncestor(const SpatialIndex& key);

  //! Count the occurences of the value.
  /*!
    We need to overide the base class function because different keys
    may have the same code.  We need to check the level as well.
  */
  bool
  count(const SpatialIndex& key) const
  {
    const_iterator i = find(key);
    return i != end() && i->first == key;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Partition.
  //@{
public:

  //! Partition the nodes.
  void
  partition(iterator* start, iterator* finish);

  //! Partition the nodes.
  void
  partition(std::vector<iterator>* delimiters);

  //! Partition the nodes with the constraint that mergeable block do not cross partitions.
  void
  partitionMergeable(std::vector<iterator>* delimiters);

};

//! Perform refinement with the supplied criterion.
/*!
  \return The number of splitting operations.
*/
template<typename _Patch, class _Traits, typename _Function>
int
refine(Orthtree<_Patch, _Traits>* orthtree, _Function refinePredicate);

//! Perform coarsening with the supplied criterion.
/*!
  \return The number of merging operations.
*/
template<typename _Patch, class _Traits, typename _Function>
int
coarsen(Orthtree<_Patch, _Traits>* orthtree, _Function coarsenPredicate,
        bool areBalancing = true);

//! Perform a single coarsening sweep with the supplied criterion.
/*!
  \return The number of merging operations.
*/
template<typename _Patch, class _Traits, typename _Function>
int
coarsenSweep(Orthtree<_Patch, _Traits>* orthtree, _Function coarsenPredicate,
             bool areBalancing = true);

//! Return true if the orthtree has nodes at the specified level.
/*!
  \relates Orthtree
*/
template<typename _Patch, class _Traits>
inline
bool
hasNodesAtLevel(const Orthtree<_Patch, _Traits>& orthtree,
                const std::size_t level)
{
  typedef Orthtree<_Patch, _Traits> Orthtree;
  typedef typename Orthtree::const_iterator const_iterator;
  for (const_iterator i = orthtree.begin(); i != orthtree.end(); ++i) {
    if (i->first.getLevel() == level) {
      return true;
    }
  }
  return false;
}

//! Print the orthtree.
/*!
  \relates Orthtree
  Don't print the ghost patches.
*/
template<typename _Patch, class _Traits>
inline
std::ostream&
operator<<(std::ostream& out, const Orthtree<_Patch, _Traits>& x)
{
  typedef Orthtree<_Patch, _Traits> Orthtree;
  for (typename Orthtree::const_iterator i = x.begin(); i != x.end(); ++i) {
    if (! x.isGhost(i)) {
      out << i->first << "\n" << i->second << "\n";
    }
  }
  return out;
}

//! Print the bounding boxes for the leaves in VTK format.
/*!
  \relates Orthtree
  Don't print anything for the ghost patches.
*/
template<typename _Patch, class _Traits>
void
printVtkUnstructuredGrid(std::ostream& out, const Orthtree<_Patch, _Traits>& x);

} // namespace amr
}

#define __amr_Orthtree_ipp__
#include "stlib/amr/Orthtree.ipp"
#undef __amr_Orthtree_ipp__

#endif
