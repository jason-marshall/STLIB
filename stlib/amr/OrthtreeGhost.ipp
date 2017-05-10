// -*- C++ -*-

#if !defined(__amr_Orthtree_ipp__)
#error This file is an implementation detail of the class Orthtree.
#endif

namespace stlib
{
namespace amr
{

//---------------------------------------------------------------------------
// Constructors etc.
//---------------------------------------------------------------------------

// Make an empty tree.
template<typename _Patch, class _Traits>
inline
Orthtree<_Patch, _Traits>::
Orthtree(const Point& lowerCorner, const Point& extents) :
  _nodes(),
  _lowerCorner(lowerCorner),
  _extents()
{
  // Compute the leaf extents for each level.
  _extents[0] = extents;
  for (std::size_t i = 1; i < _extents.size(); ++i) {
    _extents[i] = _extents[i - 1];
    _extents[i] *= 0.5;
  }
}

//---------------------------------------------------------------------------
// Accessors.
//---------------------------------------------------------------------------

// Compute the lower corner of the leaf.
template<typename _Patch, class _Traits>
inline
void
Orthtree<_Patch, _Traits>::
computeLowerCorner(const SpatialIndex& key, Point* lowerCorner) const
{
  const Point& extents = getExtents(key);
  *lowerCorner = key.getCoordinates();
  *lowerCorner *= extents;
  *lowerCorner += _lowerCorner;
}


// Get the keys that are parents of 2^Dimension leaves.
template<typename _Patch, class _Traits>
template<typename _OutputIterator>
inline
void
Orthtree<_Patch, _Traits>::
getParentKeys(_OutputIterator parentKeys) const
{
  // For each leaf.
  for (const_iterator i = begin(); i != end(); ++i) {
    // If this is a local lower corner.
    if (hasParent(i->first) && isLowerCorner(i->first)) {
      // Get the parent key.
      SpatialIndex key = i->first;
      key.transformToParent();
      // See if the sibling leaves are in the tree.
      bool siblingsArePresent = true;
      for (std::size_t i = 1; i != NumberOfOrthants; ++i) {
        key.transformToChild(i);
        if (! count(key)) {
          siblingsArePresent = false;
          break;
        }
        key.transformToParent();
      }
      if (siblingsArePresent) {
        *parentKeys++ = key;
      }
    }
  }
}


// Get the keys that are parents of 2^Dimension leaves and would result in
// a balanced tree under merging.
template<typename _Patch, class _Traits>
template<typename _OutputIterator>
inline
void
Orthtree<_Patch, _Traits>::
getParentKeysBalanced(_OutputIterator parentKeys) const
{
  // Get the parent keys.
  std::vector<SpatialIndex> keys;
  getParentKeys(std::back_inserter(keys));
  // Check which merging operations would result in a balanced tree.
  for (typename std::vector<SpatialIndex>::const_iterator i = keys.begin();
       i != keys.end(); ++i) {
    // If merging would not imbalance the tree.
    if (! needsRefinementToBalance(*i)) {
      // Output that parent key.
      *parentKeys++ = *i;
    }
  }
}


// Return true if the tree is balanced.
template<typename _Patch, class _Traits>
inline
bool
Orthtree<_Patch, _Traits>::
isBalanced() const
{
  // Check each node.
  for (const_iterator i = begin(); i != end(); ++i) {
    if (needsRefinementToBalance(i->first)) {
      return false;
    }
  }
  return true;
}


// Get the adjacent neighbors in the specified direction in a balanced tree.
template<typename _Patch, class _Traits>
template<typename _OutputIterator>
inline
void
Orthtree<_Patch, _Traits>::
getBalancedNeighbors(const const_iterator node, _OutputIterator output)
const
{
  // For each direction.
  for (std::size_t direction = 0; direction != 2 * Dimension; ++direction) {
    // Get the neighbors in that direction.
    getBalancedNeighbors(node, direction, output);
  }
}


// Get the adjacent neighbors in the specified direction in a balanced tree.
template<typename _Patch, class _Traits>
template<typename _OutputIterator>
inline
void
Orthtree<_Patch, _Traits>::
getBalancedNeighbors(const const_iterator node,
                     const std::size_t neighborDirection,
                     _OutputIterator output) const
{
  //
  // Get the key of the neighbor at the same level.
  //
  SpatialIndex key(node->first);
  // If there is no adjacent neighbor in this direction.
  if (! hasNeighbor(key, neighborDirection)) {
    // Do nothing.
    return;
  }
  key.transformToNeighbor(neighborDirection);

  // Search for the neighbor.  Find the last element that is not
  // greater than the key.
  const_iterator neighbor = findAncestor(key);

  // If there is a neighbor at the same level or one level lower.
  if (neighbor->first.getLevel() <= key.getLevel()) {
#ifdef STLIB_DEBUG
    // Check that the level differs by at most 1.
    assert(key.getLevel() - neighbor->first.getLevel() <= 1);
#endif
    *output++ = neighbor;
  }
  // The adjacent neighbors are at a higher level.
  else {
    SpatialIndex child;
    // Get the adjacent neighbors by refining the same-level neighbor key.
    // The refinement direction is the opposite of the adjacent direction.
    const std::size_t refinementDirection = neighborDirection -
                                            2 * (neighborDirection % 2) + 1;
    const std::size_t coordinate = refinementDirection / 2;
    const std::size_t direction = refinementDirection % 2;
    for (std::size_t i = 0; i != NumberOfOrthants; ++i) {
      // If this is refinement in the desired direction.
      if ((i >> coordinate) % 2 == direction) {
        child = key;
        child.transformToChild(i);
        const_iterator c = find(child);
        *output++ = c;
#ifdef STLIB_DEBUG
        assert(c != end() && c->first.getLevel() - 1 == key.getLevel());
#endif
      }
    }
  }
}


// Return true if the element needs refinement in order to balance the tree.
template<typename _Patch, class _Traits>
inline
bool
Orthtree<_Patch, _Traits>::
needsRefinementToBalance(const SpatialIndex& key) const
{
  // If there are no levels more than one higher than this node's level.
  if (key.getLevel() > MaximumLevel - 2) {
    return false;
  }
  const Level maximumAllowedLevel = key.getLevel() + 1;
  SpatialIndex neighbor, child;
  // Check for neighbors in each direction.
  for (std::size_t adjacentDirection = 0; adjacentDirection != 2 * Dimension;
       ++adjacentDirection) {
    // The neighbor at the same level.
    neighbor = key;
    if (! hasNeighbor(neighbor, adjacentDirection)) {
      continue;
    }
    neighbor.transformToNeighbor(adjacentDirection);
    // The refinement direction is the opposite of the adjacent direction.
    const std::size_t refinementDirection = adjacentDirection -
                                            2 * (adjacentDirection % 2) + 1;
    const std::size_t coordinate = refinementDirection / 2;
    const std::size_t direction = refinementDirection % 2;
    for (std::size_t i = 0; i != NumberOfOrthants; ++i) {
      if ((i >> coordinate) % 2 == direction) {
        child = neighbor;
        child.transformToChild(i, 2);
        const_iterator c = find(child);
        if (c != end() && c->first.getLevel() > maximumAllowedLevel) {
          return true;
        }
      }
    }
  }

  // We did not find any small adjacent neighbors.
  return false;
}


// Return true if the node has a higher level neighbor in the specified
// direction.
template<typename _Patch, class _Traits>
inline
bool
Orthtree<_Patch, _Traits>::
hasHigherNeighbor(const const_iterator node, const std::size_t direction)
const
{
  // If there is no neighbor in that direction.
  if (! hasNeighbor(node, direction)) {
    // Then there are no higher level neighbors in that direction.
    return false;
  }
  // Find the node that intersects that intersects the neighbor at the
  // same level.
  SpatialIndex neighborKey(node->first);
  neighborKey.transformToNeighbor(direction);
  const_iterator neighbor = findAncestor(neighborKey);
  // Check the level of that node.
  return neighbor->first.getLevel() > node->first.getLevel();
}


// Return true if the node has a higher level neighbor in the specified
// direction.
template<typename _Patch, class _Traits>
inline
bool
Orthtree<_Patch, _Traits>::
hasHigherNeighbor(const const_iterator node) const
{
  // Check each direction.
  for (std::size_t direction = 0; direction != 2 * Dimension; ++direction) {
    if (hasHigherNeighbor(node, direction)) {
      return true;
    }
  }
  return false;
}


//---------------------------------------------------------------------------
// Manipulators.
//---------------------------------------------------------------------------


// Insert a value type.
template<typename _Patch, class _Traits>
inline
typename Orthtree<_Patch, _Traits>::iterator
Orthtree<_Patch, _Traits>::
insert(const value_type& x)
{
  // Try to insert the value.
  std::pair<iterator, bool> result = _nodes.insert(x);
  // Assert that we were able to do so.
  assert(result.second);
  return result.first;
}


// Refine an element.  Get the children.
template<typename _Patch, class _Traits>
template<typename _OutputIterator>
inline
typename Orthtree<_Patch, _Traits>::iterator
Orthtree<_Patch, _Traits>::
split(const iterator parent, _OutputIterator children)
{
#ifdef STLIB_DEBUG
  // Make sure this is not a ghost patch.
  assert(! isGhost(parent));
#endif

  // Erase the parent.  We have to do this first because one of the children
  // will have the same code.
  const SpatialIndex parentKey = parent->first;
  erase(parent);

  // CONTINUE: This causes a compilation warning.
  Patch defaultPatch;

  // The first child.
  SpatialIndex key = parentKey;
  key.transformToChild(0);
  const iterator firstChild = insert(key, defaultPatch);
  *children++ = firstChild;

  // The rest of the children.
  iterator position = firstChild;
  for (std::size_t i = 1; i != NumberOfOrthants; ++i) {
    // Make the key.
    // CONTINUE: Can a transform to next instead? Would that be faster?
    SpatialIndex key = parentKey;
    key.transformToChild(i);
    // Insert the node. Use the position of the previously inserted child
    // as a hint to the insert location. (This is faster than either inserting
    // without a hint or using the position after the previously inserted
    // child as a hint.)
    *children++ = position = insert(position, key, defaultPatch);
  }
  return firstChild;
}

// CONTINUE: Can I use position hints for the inserts?
// Coarsen leaves given the parent key.  Return the coarsened leaf.
template<typename _Patch, class _Traits>
inline
typename Orthtree<_Patch, _Traits>::iterator
Orthtree<_Patch, _Traits>::
merge(iterator firstChild)
{
  // The parent key.
  SpatialIndex parentKey = firstChild->first;
  parentKey.transformToParent();

  // Erase the children.
  for (std::size_t i = 0; i != NumberOfOrthants; ++i) {
#ifdef STLIB_DEBUG
    // Make sure this child is not a ghost patch.
    assert(! isGhost(firstChild));
    assert(firstChild != end());
    assert(firstChild->first.getLevel() == parentKey.getLevel() + 1);
#endif
    // Increment the iterator before we erase the node. (Erasing invalidates
    // the iterator.)
    iterator node = firstChild;
    ++firstChild;
    erase(node);
  }
  // Insert the parent.
  return insert(parentKey);
}

// Perform refinement to balance the tree. Record the new nodes.
template<typename _Patch, class _Traits>
template<typename _OutputIterator>
inline
int
Orthtree<_Patch, _Traits>::
balance(_OutputIterator newNodes)
{
  // The set of leaves that we may refine.  We need a set instead of a vector
  // because when we refine a node we add the lower level neighbors as
  // well as the new children.  Some of the neighbors may already be in the
  // set.
  std::set<iterator, CompareIterator> mayRefine;
  std::insert_iterator<std::set<iterator, CompareIterator> >
  insertIterator(mayRefine, mayRefine.end());
  for (iterator i = begin(); i != end(); ++i) {
    if (! isGhost(i)) {
      *insertIterator++ = i;
    }
  }

  std::set<iterator, CompareIterator> newNodesSet;
  iterator first, child;
  std::size_t count = 0;
  while (! mayRefine.empty()) {
    // Get a node.
    iterator i = *--mayRefine.end();
    mayRefine.erase(--mayRefine.end());
    // If we need to refine the node.
    if (needsRefinementToBalance(i->first)) {
      // Add non-ghost neighbors to the set of nodes which may need refinement.
      getLowerNeighbors(i, insertIterator);
      // Remove from the set of new nodes (if present).
      newNodesSet.erase(i);
      // Refine the node and add the children to the set of nodes that may
      // need to be refined.
      first = split(i);
      // Add the children to the set of nodes that may need to be refined.
      child = first;
      for (std::size_t n = 0; n != NumberOfOrthants; ++n) {
        mayRefine.insert(child);
        ++child;
      }
      // Add the children to the set of new nodes.
      child = first;
      for (std::size_t n = 0; n != NumberOfOrthants; ++n) {
        newNodesSet.insert(child);
        ++child;
      }
      ++count;
    }
  }

  // Record the new nodes.
  for (typename std::set<iterator, CompareIterator>::const_iterator i =
         newNodesSet.begin(); i != newNodesSet.end(); ++i) {
    *newNodes++ = *i;
  }

  return count;
}

// Get the non-ghost, adjacent neighbors which have lower levels.
template<typename _Patch, class _Traits>
template<typename _OutputIterator>
inline
void
Orthtree<_Patch, _Traits>::
getLowerNeighbors(const iterator node, _OutputIterator i)
{
  SpatialIndex key;
  // For each neighbor direction.
  for (std::size_t n = 0; n != 2 * Dimension; ++n) {
    key = node->first;
    // If there is an adjacent neighbor in this direction.
    if (hasNeighbor(key, n)) {
      // Get the key of the neighbor at the same level.
      key.transformToNeighbor(n);
      // Search for the neighbor.  Find the last element that is not
      // greater than the key.
      iterator neighbor = findAncestor(key);
#ifdef STLIB_DEBUG
      assert(neighbor->first.getLevel() > node->first.getLevel() ||
             areAdjacent(node->first, neighbor->first));
#endif
      // If the neighbor is non-ghost and has a lower level.
      if (! isGhost(neighbor) &&
          neighbor->first.getLevel() < node->first.getLevel()) {
        // Record the neighbor.
        *i++ = neighbor;
      }
    }
  }
}


// Get the mergeable groups of 2^Dimension nodes.
template<typename _Patch, class _Traits>
template<typename _OutputIterator>
inline
void
Orthtree<_Patch, _Traits>::
getMergeableGroups(_OutputIterator lowerCornerNodes, iterator start,
                   iterator finish)
{
  // For each leaf.
  for (iterator i = start; i != finish; ++i) {
    // If this is a local lower corner.
    if (! isGhost(i) && hasParent(i->first) && isLowerCorner(i->first)) {
      iterator j = i;
      ++j;
      bool isMergeable = true;
      for (std::size_t n = 1; n != NumberOfOrthants && isMergeable; ++n, ++j) {
        // CONTINUE: I am assuming that mergeable block are not split
        // accross processes.
        assert(j != end());
        if (isGhost(j) || i->first.getLevel() != j->first.getLevel()) {
          isMergeable = false;
        }
      }
      if (isMergeable) {
        *lowerCornerNodes++ = i;
      }
    }
  }
}


// Get the mergeable groups of 2^Dimension nodes whose merging would result
// in a balanced tree.
template<typename _Patch, class _Traits>
template<typename _OutputIterator>
inline
void
Orthtree<_Patch, _Traits>::
getMergeableGroupsBalanced(_OutputIterator lowerCornerNodes, iterator start,
                           iterator finish)
{
  // Get the mergeable groups.
  std::vector<iterator> mergeable;
  getMergeableGroups(std::back_inserter(mergeable), start, finish);
  // Check which merging operations would result in a balanced tree.
  SpatialIndex key;
  for (typename std::vector<iterator>::const_iterator i = mergeable.begin();
       i != mergeable.end(); ++i) {
    key = (*i)->first;
    key.transformToParent();
    // If merging would not imbalance the tree.
    if (! needsRefinementToBalance(key)) {
      // Output that group.
      *lowerCornerNodes++ = *i;
    }
  }
}


//---------------------------------------------------------------------------
// Search.
//---------------------------------------------------------------------------

// CONTINUE: I don't think that these are correct.
// Find the node that matches the code.  If the node is not in the tree,
// find its ancestor.
template<typename _Patch, class _Traits>
inline
typename Orthtree<_Patch, _Traits>::const_iterator
Orthtree<_Patch, _Traits>::
findAncestor(const SpatialIndex& key) const
{
  // Search for the node.  Find the first key that is not less than the key.
  const_iterator node = _nodes.lower_bound(key);
  // If the node exists, the seach will find it. Otherwise if any of its
  // descendents exist, the search will find the first of those. Otherwise
  // the search will find the next node or the end.
  if (node != end()) {
    // Check the case that the node and its lower corner descendents don't
    // exist.
    if (node->first.getCode() != key.getCode() && node != begin()) {
      // Here we either found the next node or a non-lower corner descendent.
      // For the former case, we should back up to get the ancestor.
      // For the latter case, decrementing doesn't hurt. We'll return end().
      --node;
    }
    if (isAncestor(key, node->first)) {
      return node;
    }
  }
  return end();

  // Old version.
#if 0
  if (node->first.getCode() != key.getCode()) {
#ifdef STLIB_DEBUG
    assert(node != begin());
#endif
    --node;
  }
  return node;
#endif
}

// Find the node that matches the code.  If the node is not in the tree, find
// its ancestor.
template<typename _Patch, class _Traits>
inline
typename Orthtree<_Patch, _Traits>::iterator
Orthtree<_Patch, _Traits>::
findAncestor(const SpatialIndex& key)
{
  // Search for the node.  Find the first key that is not less than the key.
  iterator node = _nodes.lower_bound(key);
  // If the node exists, the seach will find it. Otherwise if any of its
  // descendents exist, the search will find the first of those. Otherwise
  // the search will find the next node or the end.
  if (node != end()) {
    // Check the case that the node and its lower corner descendents don't
    // exist.
    if (node->first.getCode() != key.getCode() && node != begin()) {
      // Here we either found the next node or a non-lower corner descendent.
      // For the former case, we should back up to get the ancestor.
      // For the latter case, decrementing doesn't hurt. We'll return end().
      --node;
    }
    if (isAncestor(key, node->first)) {
      return node;
    }
  }
  return end();

  // Old version.
#if 0
  if (node->first.getCode() != key.getCode()) {
#ifdef STLIB_DEBUG
    assert(node != begin());
#endif
    --node;
  }
  return node;
#endif
}


//---------------------------------------------------------------------------
// Partition.
//---------------------------------------------------------------------------

template<typename _Patch, class _Traits>
inline
void
Orthtree<_Patch, _Traits>::
partition(iterator* start, iterator* finish)
{
#ifdef _OPENMP
  int a, b;
  // Note: We ignore the ghost nodes.
  numerical::partitionRange(size(), omp_get_num_threads(),
                            omp_get_thread_num(), &a, &b);
  *start = begin();
  advance(start, a);
  *finish = *start;
  advance(finish, b - a);
#else
  // Serial behavior.  The range that contains all nodes.
  *start = begin();
  *finish = end();
#endif
}

template<typename _Patch, class _Traits>
inline
void
Orthtree<_Patch, _Traits>::
partition(std::vector<iterator>* delimiters)
{
#ifdef STLIB_DEBUG
#ifdef _OPENMP
  assert(! omp_in_parallel());
  assert(omp_get_max_threads() + 1 == delimiters->numElements());
#endif
#endif
  // Set the first and last delimiter.
  *delimiters->begin() = begin();
  *(delimiters->end() - 1) = end();

  const std::size_t threads = delimiters->size() - 1;
  // Provide this exit so we don't compute the size for the serial case.
  // (Computing the size has linear complexity.)
  if (threads == 1) {
    return;
  }

  // Note: We ignore the ghost nodes.
  const std::size_t numberOfNodes = size();
  for (std::size_t i = 0; i != threads - 1; ++i) {
    // The beginning of the partition.
    (*delimiters)[i + 1] = (*delimiters)[i];
    // Advance to the end of the partition.
    advance(&(*delimiters)[i + 1],
            numerical::getPartition(numberOfNodes, threads, i));
  }
}

template<typename _Patch, class _Traits>
inline
void
Orthtree<_Patch, _Traits>::
partitionMergeable(std::vector<iterator>* delimiters)
{
  // First partition.
  partition(delimiters);
  // Then move the delimiters forward to avoid breaking up mergeable blocks.
  for (std::size_t n = 1; n != delimiters->size() - 1; ++n) {
    iterator& i = (*delimiters)[n];
    while (i != end() && ! isLowerCorner(i->first)) {
      ++i;
    }
  }
}

//---------------------------------------------------------------------------
// Free functions.
//---------------------------------------------------------------------------

// CONTINUE Ghost nodes.
// Perform refinement with the supplied criterion.
// Return the number of refinement operations.
template<typename _Patch, class _Traits, typename _Function>
inline
int
refine(Orthtree<_Patch, _Traits>* orthtree, _Function refinePredicate)
{
  typedef Orthtree<_Patch, _Traits> Orthtree;
  typedef typename Orthtree::iterator iterator;

  // CONTINUE: Threaded version.
  //
  // Serial version.
  //
  std::size_t count = 0;
  // The set of leaves that we may refine.
  std::vector<iterator> mayRefine;
  // An output iterator on the leaves that we may refine.
  std::back_insert_iterator<std::vector<iterator> > outputIterator(mayRefine);
  for (iterator i = orthtree->begin(); i != orthtree->end(); ++i) {
    mayRefine.push_back(i);
  }

  while (! mayRefine.empty()) {
    iterator i = mayRefine.back();
    mayRefine.pop_back();
    if (orthtree->canBeRefined(i) && refinePredicate(i)) {
      orthtree->split(i, outputIterator);
      ++count;
    }
  }
  return count;
}


// Perform coarsening with the supplied criterion.
// Return The number of merging operations.
template<typename _Patch, class _Traits, typename _Function>
inline
int
coarsen(Orthtree<_Patch, _Traits>* orthtree, _Function coarsenPredicate,
        const bool areBalancing)
{
  std::size_t countCoarsened = 0;
  std::size_t c;
  // Perform sweeps until no more nodes are coarsened.
  do {
    c = coarsenSweep(orthtree, coarsenPredicate, areBalancing);
    countCoarsened += c;
  }
  while (c != 0);
  return countCoarsened;
}


// CONTINUE Ghost nodes.
// Perform a single coarsening sweep with the supplied criterion.
// Return The number of merging operations.
template<typename _Patch, class _Traits, typename _Function>
inline
int
coarsenSweep(Orthtree<_Patch, _Traits>* orthtree, _Function coarsenPredicate,
             bool areBalancing)
{
  typedef Orthtree<_Patch, _Traits> Orthtree;
  typedef typename Orthtree::SpatialIndex SpatialIndex;
  typedef typename Orthtree::iterator iterator;

#ifdef _OPENMP
  //
  // Threaded version.
  //
  // Partition the nodes.  We need to do this outside the concurrent block
  // because it may change the number of nodes.
  const int threads = omp_get_max_threads();
  ads::Array<1, iterator> delimiters(threads + 1);
  orthtree->partitionMergeable(&delimiters);
  ads::Array<1, iterator> upper(threads, delimiters.begin() + 1,
                                delimiters.end());
  for (std::size_t i = 0; i != upper.size() - 1; ++i) {
    --upper[i];
  }

  int countCoarsened = 0;
  #pragma omp parallel reduction (+:countCoarsened)
  {
    // Get our partition
    const int thread = omp_get_thread_num();
    const iterator start = delimiters[thread];
    const iterator finish = upper[thread];

    // The mergeable groups.
    std::vector<iterator> mergeable;
    std::back_insert_iterator<std::vector<iterator> >
    outputIterator(mergeable);
    if (areBalancing) {
      orthtree->getMergeableGroupsBalanced(outputIterator, start, finish);
    }
    else {
      orthtree->getMergeableGroups(outputIterator, start, finish);
    }
    std::vector<iterator> doMerge;
    Key key;
    for (typename std::vector<iterator>::const_iterator i = mergeable.begin();
         i != mergeable.end(); ++i) {
      // If the group should be coarsened.
      if (coarsenPredicate(*i)) {
        doMerge.push_back(*i);
      }
    }
    // Perform the merging.  This needs to be performed in a critical
    // section as it modifies the container.
    #pragma omp critical
    for (typename std::vector<iterator>::const_iterator i = doMerge.begin();
         i != doMerge.end(); ++i) {
      orthtree->merge(*i);
    }
    countCoarsened += doMerge.size();
  }
#else
  //
  // Serial version.
  //
  std::size_t countCoarsened = 0;
  // The mergeable groups.
  std::vector<iterator> mergeable;
  std::back_insert_iterator<std::vector<iterator> >
  outputIterator(mergeable);
  if (areBalancing) {
    orthtree->getMergeableGroupsBalanced(outputIterator);
  }
  else {
    orthtree->getMergeableGroups(outputIterator);
  }
  SpatialIndex key;
  for (typename std::vector<iterator>::const_iterator i = mergeable.begin();
       i != mergeable.end(); ++i) {
    // If the group should be coarsened.
    if (coarsenPredicate(*i)) {
      orthtree->merge(*i);
      ++countCoarsened;
    }
  }
#endif
  return countCoarsened;
}


//! Define the VTK output type.
/*!
  This class is a level of indirection that allows one to define supported
  output types.  By default, element data is not printed when writing output
  in VTK format.  Below we define output for \c double, \c float, and \c int.
  If a user-defined element can be printed in one of these formats,
  specialize this class with that element type.
*/
template<typename _T>
struct PatchVtkOutput {
  //! For un-supported types, the output type is void*.
  typedef void* Type;
};

//! Define the VTK output type.
template<>
struct PatchVtkOutput<double> {
  //! The output type is double.
  typedef double Type;
};

//! Define the VTK output type.
template<>
struct PatchVtkOutput<float> {
  //! The output type is float.
  typedef float Type;
};

//! Define the VTK output type.
template<>
struct PatchVtkOutput<int> {
  //! The output type is int.
  typedef int Type;
};


// Print the bounding boxes for the leaves in VTK format.
template<typename _Patch, class _Traits>
inline
void
printVtkUnstructuredGrid
(std::ostream& out,
 const Orthtree<_Patch, _Traits>& x)
{
  printVtkUnstructuredGrid(std::integral_constant<bool,
                           _Traits::Dimension <= 3>(), out, x);
}

// Print a message that VTK output in this dimension is not supported.
template<typename _Patch, class _Traits>
inline
void
printVtkUnstructuredGrid
(std::false_type /*supported dimension*/,
 std::ostream& out,
 const Orthtree<_Patch, _Traits>& x)
{
  out << _Traits::Dimension << "-D VTK output is not supported.\n";
}



template<typename _Orthtree>
class PrintElementsVtkDataArray
{
public:

  // Print the element data array.
  void
  operator()(std::ostream& out, const _Orthtree& orthtree)
  {
    typedef typename PatchVtkOutput<typename _Orthtree::Patch>::Type Type;
    Type x = Type();
    print(out, orthtree, x);
  }

private:

  // Print nothing for unsupported element types.
  template<typename _T>
  void
  print(std::ostream& out, const _Orthtree& orthtree, _T /*dummy*/)
  {
  }

  // Print the element data array.
  //template<>
  void
  print(std::ostream& out, const _Orthtree& orthtree, double /*dummy*/)
  {
    out << "<DataArray type=\"Float64\" Name=\"element\">\n";
    printTheRest(out, orthtree);
  }

  // Print the element data array.
  //template<>
  void
  print(std::ostream& out, const _Orthtree& orthtree, float /*dummy*/)
  {
    out << "<DataArray type=\"Float32\" Name=\"element\">\n";
    printTheRest(out, orthtree);
  }

  // Print the element data array.
  //template<>
  void
  print(std::ostream& out, const _Orthtree& orthtree, int /*dummy*/)
  {
    out << "<DataArray type=\"Int32\" Name=\"element\">\n";
    printTheRest(out, orthtree);
  }

  void
  printTheRest(std::ostream& out, const _Orthtree& x)
  {
    typedef typename _Orthtree::const_iterator Iterator;

    for (Iterator i = x.begin(); i != x.end(); ++i) {
      out << i->second << "\n";
    }
    out << "</DataArray>\n";
  }
};


// Print the element.
template<typename _Patch, class _Traits>
inline
void
printElementsVtkDataArray
(std::ostream& out,
 const Orthtree<_Patch, _Traits>& x)
{
  typedef Orthtree<_Patch, _Traits> Orthtree;
  PrintElementsVtkDataArray<Orthtree> printer;
  printer(out, x);
}


// Print the bounding boxes for the leaves in VTK format.
template<typename _Patch, class _Traits>
inline
void
printVtkUnstructuredGrid
(std::true_type /*supported dimension*/,
 std::ostream& out,
 const Orthtree<_Patch, _Traits>& x)
{
  typedef Orthtree<_Patch, _Traits> Orthtree;
  typedef typename Orthtree::const_iterator Iterator;
  typedef typename Orthtree::Point Point;

  static_assert(_Traits::Dimension >= 1 && _Traits::Dimension <= 3,
                "Bad dimension.");

  Point lowerCorner, p;
  const std::size_t size = x.size();

  // Header.
  out << "<?xml version=\"1.0\"?>\n";
  // Begin VTKFile.
  out << "<VTKFile type=\"UnstructuredGrid\">\n";
  // Begin UnstructuredGrid.
  out << "<UnstructuredGrid>\n";
  // Begin Piece.
  out << "<Piece NumberOfPoints=\"" << Orthtree::NumberOfOrthants* size
      << "\" NumberOfCells=\"" << size << "\">\n";

  // Begin PointData.
  out << "<PointData>\n";
  // End PointData.
  out << "</PointData>\n";

  // Begin CellData.
  out << "<CellData>\n";
  // The elements.
  printElementsVtkDataArray(out, x);
  // The level.
  out << "<DataArray type=\"Int32\" Name=\"level\">\n";
  for (Iterator i = x.begin(); i != x.end(); ++i) {
    if (! x.isGhost(i)) {
      out << std::size_t(i->first.getLevel()) << "\n";
    }
  }
  out << "</DataArray>\n";
  // The coordinates.
  for (std::size_t d = 0; d != _Traits::Dimension; ++d) {
    out << "<DataArray type=\"Int32\" Name=\"coordinate" << d << "\">\n";
    for (Iterator i = x.begin(); i != x.end(); ++i) {
      if (! x.isGhost(i)) {
        out << std::size_t(i->first.getCoordinates()[d]) << "\n";
      }
    }
    out << "</DataArray>\n";
  }
  // The rank.
  out << "<DataArray type=\"Int32\" Name=\"rank\">\n";
  for (std::size_t rank = 0; rank != size; ++rank) {
    out << rank << "\n";
  }
  out << "</DataArray>\n";
  // End CellData.
  out << "</CellData>\n";

  // Begin Points.
  out << "<Points>\n";
  out << "<DataArray type=\"Float32\" NumberOfComponents=\"3\">\n";
  for (Iterator i = x.begin(); i != x.end(); ++i) {
    if (x.isGhost(i)) {
      continue;
    }
    const Point& extents = x.getExtents(i->first);
    x.computeLowerCorner(i->first, &lowerCorner);
    if (_Traits::Dimension == 1) {
      out << lowerCorner << " 0 0\n";
      p = lowerCorner;
      p[0] += extents[0];
      out << p << " 0 0\n";
    }
    else if (_Traits::Dimension == 2) {
      out << lowerCorner << " 0\n";
      p = lowerCorner;
      p[0] += extents[0];
      out << p << " 0\n";
      out << lowerCorner + extents << " 0\n";
      p = lowerCorner;
      p[1] += extents[1];
      out << p << " 0\n";
    }
    else if (_Traits::Dimension == 3) {
      // 0
      out << lowerCorner << "\n";
      // 1
      p = lowerCorner;
      p[0] += extents[0];
      out << p << "\n";
      // 2
      p[1] += extents[1];
      out << p << "\n";
      // 3
      p[0] -= extents[0];
      out << p << "\n";
      // 4
      p = lowerCorner;
      p[2] += extents[2];
      out << p << "\n";
      // 5
      p[0] += extents[0];
      out << p << "\n";
      // 6
      p[1] += extents[1];
      out << p << "\n";
      // 7
      p[0] -= extents[0];
      out << p << "\n";
    }
    else {
      assert(false);
    }
  }
  out << "</DataArray>\n";
  // End Points.
  out << "</Points>\n";

  // Begin Cells.
  out << "<Cells>\n";
  out << "<DataArray type=\"Int32\" Name=\"connectivity\">\n";
  for (std::size_t n = 0; n != size; ++n) {
    for (std::size_t i = 0; i != Orthtree::NumberOfOrthants; ++i) {
      out << Orthtree::NumberOfOrthants* n + i << " ";
    }
    out << "\n";
  }
  out << "</DataArray>\n";
  out << "<DataArray type=\"Int32\" Name=\"offsets\">\n";
  for (std::size_t n = 1; n <= size; ++n) {
    out << Orthtree::NumberOfOrthants* n << "\n";
  }
  out << "</DataArray>\n";
  out << "<DataArray type=\"UInt8\" Name=\"types\">\n";
  for (std::size_t n = 0; n != size; ++n) {
    if (_Traits::Dimension == 1) {
      // Each cell is a line.
      out << "3\n";
    }
    else if (_Traits::Dimension == 2) {
      // Each cell is a quad.
      out << "9\n";
    }
    else if (_Traits::Dimension == 3) {
      // Each cell is a hexahedron.
      out << "12\n";
    }
    else {
      assert(false);
    }
  }
  out << "</DataArray>\n";
  // End Cells.
  out << "</Cells>\n";

  // End Piece.
  out << "</Piece>\n";

  // End UnstructuredGrid.
  out << "</UnstructuredGrid>\n";

  // End VTKFile.
  out << "</VTKFile>\n";
}

} // namespace amr
}
