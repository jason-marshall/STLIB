// -*- C++ -*-

#if !defined(__GridBase_ipp__)
#error This file is an implementation detail of the class GridBase
#endif

namespace stlib
{
namespace cpt
{

//
// Constructors, Destructor
//

template<std::size_t N, typename T>
inline
GridBase<N, T>&
GridBase<N, T>::
operator=(const GridBase& other)
{
  if (this != &other) {
    _distance = other._distance;
    _gradientOfDistance = other._gradientOfDistance;
    _closestPoint = other._closestPoint;
    _closestFace = other._closestFace;
    _distanceExternal = other._distanceExternal;
    _gradientOfDistanceExternal = other._gradientOfDistanceExternal;
    _closestPointExternal = other._closestPointExternal;
    _closestFaceExternal = other._closestFaceExternal;
  }
  return *this;
}

//
// Mathematical operations
//


// Calculate the signed distance, closest point, etc. for the specified
// grid points.
template<std::size_t N, typename T>
template<class Component>
inline
std::pair<std::size_t, std::size_t>
GridBase<N, T>::
computeClosestPointTransform(const std::vector<IndexList>& indices,
                             const std::vector<Point>& positions,
                             const Component& component,
                             const Number maximumDistance)
{
  // Sanity check.
  assert(indices.size() == positions.size());

  // Variables used in the following loops.
  std::size_t numberOfDistancesComputed = 0;
  std::size_t numberOfDistancesSet = 0;
  Index index;
  Number dist;

  // Iterators for the loop over the grid points.
  typename std::vector<IndexList>::const_iterator indexIterator
    = indices.begin();
  const typename std::vector<IndexList>::const_iterator indexEnd
    = indices.end();
  typename std::vector<Point>::const_iterator pointIterator
    = positions.begin();

  if (isClosestPointBeingComputed() && isGradientOfDistanceBeingComputed() &&
      isClosestFaceBeingComputed()) {
    Point cp, grad;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance, gradient of distance and closest point.
        dist = component.computeClosestPointAndGradient
               (*pointIterator, &cp, &grad);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (std::abs(dist) <= maximumDistance &&
            std::abs(dist) < std::abs(getDistance()[index])) {
          // If this is the first time the distance has been set.
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getClosestPoint()[index] = cp;
          getGradientOfDistance()[index] = grad;
          getClosestFace()[index] = component.getFaceIndex();
        }
      }
    }
  }
  else if (isClosestPointBeingComputed() &&
           isGradientOfDistanceBeingComputed()) {
    Point cp, grad;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance, gradient of distance and closest point.
        dist = component.computeClosestPointAndGradient
               (*pointIterator, &cp, &grad);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (std::abs(dist) <= maximumDistance &&
            std::abs(dist) < std::abs(getDistance()[index])) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getClosestPoint()[index] = cp;
          getGradientOfDistance()[index] = grad;
        }
      }
    }
  }
  else if (isClosestPointBeingComputed() && isClosestFaceBeingComputed()) {
    Point cp;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance and closest point.
        dist = component.computeClosestPoint(*pointIterator, &cp);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (std::abs(dist) <= maximumDistance &&
            std::abs(dist) < std::abs(getDistance()[index])) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getClosestPoint()[index] = cp;
          getClosestFace()[index] = component.getFaceIndex();
        }
      }
    }
  }
  else if (isGradientOfDistanceBeingComputed() && isClosestFaceBeingComputed()) {
    Point grad;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance and the gradient of the distance.
        dist = component.computeGradient(*pointIterator, &grad);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (std::abs(dist) <= maximumDistance &&
            std::abs(dist) < std::abs(getDistance()[index])) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getGradientOfDistance()[index] = grad;
          getClosestFace()[index] = component.getFaceIndex();
        }
      }
    }
  }
  else if (isClosestPointBeingComputed()) {
    Point cp;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance and the closest point.
        dist = component.computeClosestPoint(*pointIterator, &cp);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (std::abs(dist) <= maximumDistance &&
            std::abs(dist) < std::abs(getDistance()[index])) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getClosestPoint()[index] = cp;
        }
      }
    }
  }
  else if (isGradientOfDistanceBeingComputed()) {
    Point grad;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance and the gradient of the distance.
        dist = component.computeGradient(*pointIterator, &grad);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (std::abs(dist) <= maximumDistance &&
            std::abs(dist) < std::abs(getDistance()[index])) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getGradientOfDistance()[index] = grad;
        }
      }
    }
  }
  else if (isClosestFaceBeingComputed()) {
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance.
        dist = component.computeDistance(*pointIterator);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (std::abs(dist) <= maximumDistance &&
            std::abs(dist) < std::abs(getDistance()[index])) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getClosestFace()[index] = component.getFaceIndex();
        }
      }
    }
  }
  else {
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      // CONTINUE: This could be the problem.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance.
        dist = component.computeDistance(*pointIterator);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (std::abs(dist) <= maximumDistance &&
            std::abs(dist) < std::abs(getDistance()[index])) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
        }
      }
    }
  }

  copyToExternal();
  return std::pair<std::size_t, std::size_t>(numberOfDistancesComputed,
         numberOfDistancesSet);
}





// Calculate the signed distance, closest point, etc. for the specified
// grid points.
template<std::size_t N, typename T>
template<class Component>
inline
std::pair<std::size_t, std::size_t>
GridBase<N, T>::
computeClosestPointTransformUnsigned(const std::vector<IndexList>& indices,
                                     const std::vector<Point>& positions,
                                     const Component& component,
                                     const Number maximumDistance)
{
  // Sanity check.
  assert(indices.size() == positions.size());

  // Variables used in the following loops.
  std::size_t numberOfDistancesComputed = 0;
  std::size_t numberOfDistancesSet = 0;
  Index index;
  Number dist;

  // Iterators for the loop over the grid points.
  typename std::vector<IndexList>::const_iterator indexIterator
    = indices.begin();
  const typename std::vector<IndexList>::const_iterator indexEnd
    = indices.end();
  typename std::vector<Point>::const_iterator pointIterator
    = positions.begin();

  if (isClosestPointBeingComputed() && isGradientOfDistanceBeingComputed() &&
      isClosestFaceBeingComputed()) {
    Point cp, grad;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance, gradient of distance and closest point.
        dist = component.computeClosestPointAndGradientUnsigned
               (*pointIterator, &cp, &grad);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (dist <= maximumDistance && dist < getDistance()[index]) {
          // CONTINUE: Is this really what I want to count?
          // Should I instead compute the total number of distance computed,
          // (as opposed to distances set).
          // If this is the first time the distance has been set.
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getClosestPoint()[index] = cp;
          getGradientOfDistance()[index] = grad;
          getClosestFace()[index] = component.getFaceIndex();
        }
      }
    }
  }
  else if (isClosestPointBeingComputed() &&
           isGradientOfDistanceBeingComputed()) {
    Point cp, grad;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance, gradient of distance and closest point.
        dist = component.computeClosestPointAndGradientUnsigned
               (*pointIterator, &cp, &grad);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (dist <= maximumDistance && dist < getDistance()[index]) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getClosestPoint()[index] = cp;
          getGradientOfDistance()[index] = grad;
        }
      }
    }
  }
  else if (isClosestPointBeingComputed() && isClosestFaceBeingComputed()) {
    Point cp;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance and closest point.
        dist = component.computeClosestPointUnsigned(*pointIterator, &cp);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (dist <= maximumDistance && dist < getDistance()[index]) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getClosestPoint()[index] = cp;
          getClosestFace()[index] = component.getFaceIndex();
        }
      }
    }
  }
  else if (isGradientOfDistanceBeingComputed() &&
           isClosestFaceBeingComputed()) {
    Point grad;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance and the gradient of the distance.
        dist = component.computeGradientUnsigned(*pointIterator, &grad);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (dist <= maximumDistance && dist < getDistance()[index]) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getGradientOfDistance()[index] = grad;
          getClosestFace()[index] = component.getFaceIndex();
        }
      }
    }
  }
  else if (isClosestPointBeingComputed()) {
    Point cp;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance and the closest point.
        dist = component.computeClosestPointUnsigned(*pointIterator, &cp);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (dist <= maximumDistance && dist < getDistance()[index]) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getClosestPoint()[index] = cp;
        }
      }
    }
  }
  else if (isGradientOfDistanceBeingComputed()) {
    Point grad;
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance and the gradient of the distance.
        dist = component.computeGradientUnsigned(*pointIterator, &grad);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (dist <= maximumDistance && dist < getDistance()[index]) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getGradientOfDistance()[index] = grad;
        }
      }
    }
  }
  else if (isClosestFaceBeingComputed()) {
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance.
        dist = component.computeDistanceUnsigned(*pointIterator);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (dist <= maximumDistance && dist < getDistance()[index]) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
          getClosestFace()[index] = component.getFaceIndex();
        }
      }
    }
  }
  else {
    // Loop over the grid points.
    for (; indexIterator != indexEnd; ++indexIterator, ++pointIterator) {
      // If the index is in the index range of this grid.
      if (isIn(getRanges(), *indexIterator)) {
        // Convert the multi-index to a single container index.
        index = getDistance().arrayIndex(*indexIterator);
        // Compute the distance.
        dist = component.computeDistanceUnsigned(*pointIterator);
        ++numberOfDistancesComputed;

        // If the new distance is less than the old distance.
        if (dist <= maximumDistance && dist < getDistance()[index]) {
          if (getDistance()[index] ==
              std::numeric_limits<Number>::max()) {
            ++numberOfDistancesSet;
          }
          getDistance()[index] = dist;
        }
      }
    }
  }

  copyToExternal();
  return std::pair<std::size_t, std::size_t>(numberOfDistancesComputed,
         numberOfDistancesSet);
}





// Calculate the signed distance, closest point, etc. for the specified
// grid points.
template<std::size_t N, typename T>
template<class Component>
inline
std::pair<std::size_t, std::size_t>
GridBase<N, T>::
computeClosestPointTransform(const Lattice& lattice,
                             const Range& indexRangeInLattice,
                             const Component& component,
                             const Number maximumDistance)
{
  // Compute the intersection of the index range in the lattice and the
  // index range of this grid.
  Range indexRange = container::overlap(indexRangeInLattice, getRanges());

  // Variables used in the following loops.
  std::size_t numberOfDistancesComputed = 0;
  std::size_t numberOfDistancesSet = 0;
  Index index;
  Number dist;
  Point x;

  // Iterators over the index range.
  MultiIndexRangeIterator i = MultiIndexRangeIterator::begin(indexRange);
  const MultiIndexRangeIterator iEnd = MultiIndexRangeIterator::end(indexRange);

  if (isClosestPointBeingComputed() && isGradientOfDistanceBeingComputed() &&
      isClosestFaceBeingComputed()) {
    Point cp, grad;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance, gradient of distance and closest point.
      dist = component.computeClosestPointAndGradientChecked(x, &cp, &grad);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        // If this is the first time the distance has been set.
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getClosestPoint()[index] = cp;
        getGradientOfDistance()[index] = grad;
        getClosestFace()[index] = component.getFaceIndex();
      }
    }
  }
  else if (isClosestPointBeingComputed() &&
           isGradientOfDistanceBeingComputed()) {
    Point cp, grad;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance, gradient of distance and closest point.
      dist = component.computeClosestPointAndGradientChecked(x, &cp, &grad);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getClosestPoint()[index] = cp;
        getGradientOfDistance()[index] = grad;
      }
    }
  }
  else if (isClosestPointBeingComputed() && isClosestFaceBeingComputed()) {
    Point cp;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance and closest point.
      dist = component.computeClosestPointChecked(x, &cp);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getClosestPoint()[index] = cp;
        getClosestFace()[index] = component.getFaceIndex();
      }
    }
  }
  else if (isGradientOfDistanceBeingComputed() && isClosestFaceBeingComputed()) {
    Point grad;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance and the gradient of the distance.
      dist = component.computeGradientChecked(x, &grad);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getGradientOfDistance()[index] = grad;
        getClosestFace()[index] = component.getFaceIndex();
      }
    }
  }
  else if (isClosestPointBeingComputed()) {
    Point cp;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance and the closest point.
      dist = component.computeClosestPointChecked(x, &cp);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getClosestPoint()[index] = cp;
      }
    }
  }
  else if (isGradientOfDistanceBeingComputed()) {
    Point grad;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance and the gradient of the distance.
      dist = component.computeGradientChecked(x, &grad);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getGradientOfDistance()[index] = grad;
      }
    }
  }
  else if (isClosestFaceBeingComputed()) {
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance.
      dist = component.computeDistanceChecked(x);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getClosestFace()[index] = component.getFaceIndex();
      }
    }
  }
  else {
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance.
      dist = component.computeDistanceChecked(x);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
      }
    }
  }

  copyToExternal();
  return std::pair<std::size_t, std::size_t>(numberOfDistancesComputed,
         numberOfDistancesSet);
}









// Calculate the signed distance, closest point, etc. for the specified
// grid points.
template<std::size_t N, typename T>
template<class Component>
inline
std::pair<std::size_t, std::size_t>
GridBase<N, T>::
computeClosestPointTransformUnsigned(const Lattice& lattice,
                                     const Range& indexRangeInLattice,
                                     const Component& component,
                                     const Number maximumDistance)
{
  // Compute the intersection of the index range in the lattice and the
  // index range of this grid.
  Range indexRange = container::overlap(indexRangeInLattice, getRanges());

  // Variables used in the following loops.
  std::size_t numberOfDistancesComputed = 0;
  std::size_t numberOfDistancesSet = 0;
  Index index;
  Number dist;
  Point x;

  // Iterators over the index range.
  MultiIndexRangeIterator i = MultiIndexRangeIterator::begin(indexRange);
  const MultiIndexRangeIterator iEnd = MultiIndexRangeIterator::end(indexRange);

  if (isClosestPointBeingComputed() && isGradientOfDistanceBeingComputed() &&
      isClosestFaceBeingComputed()) {
    Point cp, grad;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance, gradient of distance and closest point.
      dist = component.computeClosestPointAndGradientUnsignedChecked
             (x, &cp, &grad);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        // If this is the first time the distance has been set.
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getClosestPoint()[index] = cp;
        getGradientOfDistance()[index] = grad;
        getClosestFace()[index] = component.getFaceIndex();
      }
    }
  }
  else if (isClosestPointBeingComputed() && isGradientOfDistanceBeingComputed()) {
    Point cp, grad;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance, gradient of distance and closest point.
      dist = component.computeClosestPointAndGradientUnsignedChecked
             (x, &cp, &grad);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getClosestPoint()[index] = cp;
        getGradientOfDistance()[index] = grad;
      }
    }
  }
  else if (isClosestPointBeingComputed() && isClosestFaceBeingComputed()) {
    Point cp;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance and closest point.
      dist = component.computeClosestPointUnsignedChecked(x, &cp);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getClosestPoint()[index] = cp;
        getClosestFace()[index] = component.getFaceIndex();
      }
    }
  }
  else if (isGradientOfDistanceBeingComputed() &&
           isClosestFaceBeingComputed()) {
    Point grad;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance and the gradient of the distance.
      dist = component.computeGradientUnsignedChecked(x, &grad);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getGradientOfDistance()[index] = grad;
        getClosestFace()[index] = component.getFaceIndex();
      }
    }
  }
  else if (isClosestPointBeingComputed()) {
    Point cp;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance and the closest point.
      dist = component.computeClosestPointUnsignedChecked(x, &cp);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getClosestPoint()[index] = cp;
      }
    }
  }
  else if (isGradientOfDistanceBeingComputed()) {
    Point grad;
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance and the gradient of the distance.
      dist = component.computeGradientUnsignedChecked(x, &grad);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getGradientOfDistance()[index] = grad;
      }
    }
  }
  else if (isClosestFaceBeingComputed()) {
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance.
      dist = component.computeDistanceUnsignedChecked(x);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
        getClosestFace()[index] = component.getFaceIndex();
      }
    }
  }
  else {
    // Loop over the grid points.
    for (; i != iEnd; ++i) {
      // Compute the position of the point.
      for (std::size_t n = 0; n != N; ++n) {
        x[n] = (*i)[n];
      }
      lattice.convertIndexToLocation(&x);
      // Convert the multi-index to a single container index.
      index = getDistance().arrayIndex(*i);
      // Compute the distance.
      dist = component.computeDistanceUnsignedChecked(x);
      ++numberOfDistancesComputed;

      // If the new distance is less than the old distance.
      if (std::abs(dist) <= maximumDistance &&
          std::abs(dist) < std::abs(getDistance()[index])) {
        if (getDistance()[index] ==
            std::numeric_limits<Number>::max()) {
          ++numberOfDistancesSet;
        }
        getDistance()[index] = dist;
      }
    }
  }

  copyToExternal();
  return std::pair<std::size_t, std::size_t>(numberOfDistancesComputed,
         numberOfDistancesSet);
}







//
// Set all the distances to std::numeric_limits<Number>::max() in
// preparation for the distance to be
// computed.  Set the gradient of the distance and the closest points to
// std::numeric_limits<Number>::max().  Set the closest faces to
// std::numeric_limits<std::size_t>::max().
//
template<std::size_t N, typename T>
inline
void
GridBase<N, T>::
initialize()
{
  // Distance.
  std::fill(_distance.begin(), _distance.end(),
            std::numeric_limits<Number>::max());
  // Gradient of distance.
  const Point infinity =
    ext::filled_array<Point>(std::numeric_limits<Number>::max());
  std::fill(_gradientOfDistance.begin(), _gradientOfDistance.end(), infinity);
  // Closest point.
  std::fill(_closestPoint.begin(), _closestPoint.end(), infinity);
  // Closest face.
  std::fill(_closestFace.begin(), _closestFace.end(),
            std::numeric_limits<std::size_t>::max());
}


template<std::size_t N, typename T>
inline
bool
GridBase<N, T>::
floodFillUnsigned(const Number farAway)
{
  bool result = false;
  //
  // Flood fill the unknown distances with farAway.
  //
  typename container::MultiArray<Number, N>::iterator i = getDistance().begin(),
                                                      iEnd = getDistance().end();
  for (; i != iEnd; ++i) {
    if (*i == std::numeric_limits<Number>::max()) {
      *i = farAway;
    }
    else {
      result = true;
    }
  }

  copyToDistanceExternal();
  return result;
}


template<std::size_t N, typename T>
inline
void
GridBase<N, T>::
copyToDistanceExternal() const
{
  if (_distanceExternal) {
    Number* out = _distanceExternal;
    for (typename container::MultiArray<Number, N>::const_iterator
         i = _distance.begin(); i != _distance.end(); ++i) {
      *out++ = *i;
    }
  }
}

template<std::size_t N, typename T>
inline
void
GridBase<N, T>::
copyToExternal() const
{
  copyToDistanceExternal();
  if (_gradientOfDistanceExternal) {
    Number* out = _gradientOfDistanceExternal;
    for (typename container::MultiArray<Point, N>::const_iterator
         i = _gradientOfDistance.begin(); i != _gradientOfDistance.end();
         ++i) {
      for (std::size_t n = 0; n != N; ++n) {
        *out++ = (*i)[n];
      }
    }
  }
  if (_closestPointExternal) {
    Number* out = _closestPointExternal;
    for (typename container::MultiArray<Point, N>::const_iterator
         i = _closestPoint.begin(); i != _closestPoint.end(); ++i) {
      for (std::size_t n = 0; n != N; ++n) {
        *out++ = (*i)[n];
      }
    }
  }
  if (_closestFaceExternal) {
    int* out = _closestFaceExternal;
    for (typename container::MultiArray<std::size_t, N>::const_iterator
         i = _closestFace.begin(); i != _closestFace.end(); ++i) {
      *out++ = *i;
    }
  }
}

//
// File I/O
//


template<std::size_t N, typename T>
inline
void
GridBase<N, T>::
put(std::ostream& out) const
{
  out << "getRanges() = " << getRanges() << '\n';
  for (auto const& x: _distance) {
    out << x << '\n';
  }
  for (auto const& x: _gradientOfDistance) {
    out << x << '\n';
  }
  for (auto const& x: _closestPoint) {
    out << x << '\n';
  }
  for (auto const& x: _closestFace) {
    out << x << '\n';
  }
}


template<std::size_t N, typename T>
inline
void
GridBase<N, T>::
displayInformation(std::ostream& out) const
{
  out << "The grid index ranges are " << getRanges() << '\n';

  if (isGradientOfDistanceBeingComputed()) {
    out << "The gradient of the distance is being computed.\n";
  }
  else {
    out << "The gradient of the distance is not being computed.\n";
  }

  if (isClosestPointBeingComputed()) {
    out << "The closest point is being computed.\n";
  }
  else {
    out << "The closest point is not being computed.\n";
  }

  if (isClosestFaceBeingComputed()) {
    out << "The closest face is being computed.\n";
  }
  else {
    out << "The closest face is not being computed.\n";
  }
}


template<std::size_t N, typename T>
inline
std::size_t
GridBase<N, T>::
countKnownDistances(const Number maximumDistance) const
{
  //
  // Find the number of known distances.
  //
  std::size_t numberOfKnownDistances = 0;
  typename container::MultiArray<Number, N>::const_iterator
  ptr = _distance.begin();
  const typename container::MultiArray<Number, N>::const_iterator
  end = _distance.end();
  for (; ptr != end; ++ptr) {
    if (std::abs(*ptr) < maximumDistance) {
      ++numberOfKnownDistances;
    }
  }

  return numberOfKnownDistances;
}


template<std::size_t N, typename T>
inline
void
GridBase<N, T>::
computeMinimumAndMaximumDistances(const Number maximumDistance,
                                  Number* minimum,
                                  Number* maximum) const
{
  //
  // Find the max and min known distance.
  //
  typename container::MultiArray<Number, N>::const_iterator
  ptr = _distance.begin();
  const typename container::MultiArray<Number, N>::const_iterator
  end = _distance.end();
  *minimum = std::numeric_limits<Number>::max();
  *maximum = -std::numeric_limits<Number>::max();
  for (; ptr != end; ++ptr) {
    if (std::abs(*ptr) < maximumDistance) {
      if (*ptr < *minimum) {
        *minimum = *ptr;
      }
      else if (*ptr > *maximum) {
        *maximum = *ptr;
      }
    }
  }
}


} // namespace cpt
}

//
// Equality operators
//


template<std::size_t N, typename T>
inline
bool
operator==(const stlib::cpt::GridBase<N, T>& a,
           const stlib::cpt::GridBase<N, T>& b)
{
  if (a.getDistance() != b.getDistance()) {
    return false;
  }

  // Check that the gradient of the distance is or is not being computed.
  if (a.isGradientOfDistanceBeingComputed() !=
      b.isGradientOfDistanceBeingComputed()) {
    return false;
  }

  // Check equality for each of the gradients of the distance.
  if (a.isGradientOfDistanceBeingComputed()) {
    if (a.getGradientOfDistance() != b.getGradientOfDistance()) {
      return false;
    }
  }

  // Check that the closest point is or is not being computed.
  if (a.isClosestPointBeingComputed() != b.isClosestPointBeingComputed()) {
    return false;
  }

  // Check equality for each of the closest points.
  if (a.isClosestPointBeingComputed()) {
    if (a.getClosestPoint() != b.getClosestPoint()) {
      return false;
    }
  }

  // Check that the closest face is or is not being computed.
  if (a.isClosestFaceBeingComputed() != b.isClosestFaceBeingComputed()) {
    return false;
  }
  // Check equality for each of the closest faces.
  if (a.isClosestFaceBeingComputed()) {
    if (a.getClosestFace() != b.getClosestFace()) {
      return false;
    }
  }

  return true;
}
