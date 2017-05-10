// -*- C++ -*-

#if !defined(__partition_BspTree_ipp__)
#error This file is an implementation detail of BspTree.
#endif

namespace stlib
{
namespace concurrent
{

//---------------------------------------------------------------------------
// 1-D
//---------------------------------------------------------------------------

template<typename _T>
inline
void
partitionRegularGridWithBspTree
(const container::MultiArrayConstRef<_T, 1>& costs,
 container::MultiArrayRef<std::size_t, 1>* identifiers,
 const _T totalCost, geom::SemiOpenInterval<1, std::ptrdiff_t> indexRange,
 std::size_t identifiersBegin, std::size_t identifiersEnd)
{
  typedef typename container::MultiArrayConstRef<_T, 1>::IndexList IndexList;

  // If we are down to a single piece or there are no elements.
  if (identifiersEnd - identifiersBegin <= 1 || indexRange.isEmpty()) {
    // Do nothing.
    return;
  }

  const std::size_t identifiersMiddle = (identifiersBegin + identifiersEnd) / 2;
  const _T splittingCost = totalCost * (identifiersMiddle - identifiersBegin) /
                           _T(identifiersEnd - identifiersBegin);

  // Find the splitting index.
  const IndexList begin = indexRange.getLowerCorner();
  const IndexList end = indexRange.getUpperCorner();
  _T cost = 0;
  IndexList splittingIndex = begin;
  for (; splittingIndex != end; ++splittingIndex[0]) {
    // If we should put this element in the lower partition.
    if (cost + costs(splittingIndex) / 2 < splittingCost) {
      // Place the element in the lower partition.
      (*identifiers)(splittingIndex) = identifiersMiddle - 1;
      cost += costs(splittingIndex);
    }
    else {
      // Stop partitioning.
      break;
    }
  }

  // Recurse.
  const std::ptrdiff_t upper = indexRange.getUpperCorner()[0];
  indexRange.setUpperCoordinate(0, splittingIndex[0]);
  partitionRegularGridWithBspTree(costs, identifiers, cost, indexRange,
                                  identifiersBegin, identifiersMiddle);
  indexRange.setLowerCoordinate(0, splittingIndex[0]);
  indexRange.setUpperCoordinate(0, upper);
  partitionRegularGridWithBspTree(costs, identifiers, totalCost - cost,
                                  indexRange, identifiersMiddle,
                                  identifiersEnd);
}



template<typename _T>
inline
void
partitionRegularGridWithBspTree
(const container::MultiArrayConstRef<_T, 1>& costs,
 container::MultiArrayRef<std::size_t, 1>* identifiers,
 std::size_t numberOfPartitions)
{
  assert(costs.range() == identifiers->range());
  // Initialize by putting all of the elements in the final partition.
  std::fill(identifiers->begin(), identifiers->end(), numberOfPartitions - 1);
  geom::SemiOpenInterval<1, std::ptrdiff_t>
  indexRange(costs.bases(), costs.bases() +
             ext::convert_array<std::ptrdiff_t>(costs.extents()));
  partitionRegularGridWithBspTree(costs, identifiers,
                                  std::accumulate(costs.begin(), costs.end(),
                                      _T(0)),
                                  indexRange, 0, numberOfPartitions);
}


//---------------------------------------------------------------------------
// 2-D
//---------------------------------------------------------------------------

inline
double
predictBestSplitting(const std::size_t n, double x, double y, double remains)
{
  if (n == 1) {
    // No communication costs.
    return 0;
  }
  if (y > x) {
    std::swap(x, y);
  }
  if (y > remains) {
    return 2 * y;
  }
  double value;
  double minValue = std::numeric_limits<double>::max();
  for (std::size_t m = n / 2; m != 0; --m) {
    // The left part is no more expensive than the right, so the left can
    // use up to half the remaining value.
    value = y + predictBestSplitting(m, m * x / n, y, (remains - y) / 2);
    value += predictBestSplitting(n - m, (n - m) * x / n, y, remains - value);
    if (value < minValue) {
      minValue = value;
    }
  }
  return minValue;
}

inline
double
predictBestSplitting(std::size_t* splittingIndex, const int n, double x,
                     double y)
{
  assert(n > 1);
  if (y > x) {
    std::swap(x, y);
  }
  double value;
  double minValue = std::numeric_limits<double>::max();
  for (std::size_t m = n / 2; m != 0; --m) {
    value = y + predictBestSplitting(m, m * x / n, y, (minValue - y) / 2);
    value += predictBestSplitting(n - m, (n - m) * x / n, y, minValue - value);
    if (value < minValue) {
      minValue = value;
      *splittingIndex = m;
    }
  }
  return minValue;
}


inline
std::size_t
predictBestSplitting(const std::size_t n, const double x, const double y)
{
  std::size_t splittingIndex = 0;
  predictBestSplitting(&splittingIndex, n, x, y);
  return splittingIndex;
}


// Reduce the index range to tightly contain the specified value.
inline
void
tightenIndexRange(const container::MultiArrayConstRef<std::size_t, 2>&
                  identifiers,
                  geom::SemiOpenInterval<2, std::ptrdiff_t>* indexRange,
                  const std::size_t value)
{
  typedef container::MultiArrayConstRef<std::size_t, 2>::IndexList
  IndexList;
  IndexList index;
  bool hasValue;

  // For each dimension.
  for (std::size_t d = 0; d != 2; ++d) {
    const std::size_t i = d;
    const std::size_t j = (d + 1) % 2;
    // Lower.
    do {
      hasValue = false;
      index[i] = indexRange->getLowerCorner()[i];
      for (index[j] = indexRange->getLowerCorner()[j];
           index[j] != indexRange->getUpperCorner()[j] && ! hasValue;
           ++index[j]) {
        if (identifiers(index) == value) {
          hasValue = true;
        }
      }
      if (! hasValue) {
        indexRange->setLowerCoordinate(i, index[i] + 1);
      }
      if (indexRange->isEmpty()) {
        return;
      }
    }
    while (! hasValue);

    // Upper.
    do {
      hasValue = false;
      index[i] = indexRange->getUpperCorner()[i] - 1;
      for (index[j] = indexRange->getLowerCorner()[j];
           index[j] != indexRange->getUpperCorner()[j] && ! hasValue;
           ++index[j]) {
        if (identifiers(index) == value) {
          hasValue = true;
        }
      }
      if (! hasValue) {
        indexRange->setUpperCoordinate(i, index[i]);
      }
      if (indexRange->isEmpty()) {
        return;
      }
    }
    while (! hasValue);
  }
}




template<typename _T>
inline
void
partitionRegularGridWithBspTree
(const container::MultiArrayConstRef<_T, 2>& costs,
 container::MultiArrayRef<std::size_t, 2>* identifiers,
 const _T totalCost, geom::SemiOpenInterval<2, std::ptrdiff_t> indexRange,
 const std::size_t identifiersBegin, const std::size_t identifiersEnd,
 const std::size_t predictionThreshhold)
{
  typedef typename container::MultiArrayConstRef<std::size_t, 2>::SizeList
  SizeList;
  typedef typename container::MultiArrayConstRef<std::size_t, 2>::IndexList
  IndexList;

  // If we are down to a single piece or there are no elements.
  if (identifiersEnd - identifiersBegin <= 1 || indexRange.isEmpty()) {
    // Do nothing.
    return;
  }

  // See if we can tighten the index range.
  tightenIndexRange(*identifiers, &indexRange, identifiersEnd - 1);
  // If there are one or fewer elements.
  if (indexRange.computeContent() <= 1) {
    return;
  }

  // There are at least two elements to partition.

  // Choose the splitting dimension.
  const SizeList extents =
    ext::convert_array<std::size_t>(indexRange.getUpperCorner() -
                                    indexRange.getLowerCorner());
  const std::size_t i = std::max_element(extents.begin(), extents.end()) -
                        extents.begin();
  const std::size_t j = (i + 1) % 2;

  // CONTINUE:
  //const int identifiersMiddle = (identifiersBegin + identifiersEnd) / 2;
  std::size_t identifiersMiddle;
  if (identifiersEnd - identifiersBegin <= predictionThreshhold) {
    identifiersMiddle = identifiersBegin +
                        predictBestSplitting(identifiersEnd - identifiersBegin,
                            extents[0], extents[1]);
  }
  else {
    identifiersMiddle = (identifiersBegin + identifiersEnd) / 2;
  }

  const _T splittingCost = totalCost * (identifiersMiddle - identifiersBegin) /
                           _T(identifiersEnd - identifiersBegin);

  // Split.
  std::size_t splittingIndex = indexRange.getUpperCorner()[i];
  _T cost = 0;
  bool stop = false;
  IndexList index;
  const IndexList begin = {{
      indexRange.getLowerCorner()[j],
      indexRange.getUpperCorner()[j] - 1
    }
  };
  const IndexList end = {{
      indexRange.getUpperCorner()[j],
      indexRange.getLowerCorner()[j] - 1
    }
  };
  const IndexList stride = {{1, -1}};
  bool flip = ((*identifiers)(indexRange.getLowerCorner()) !=
               identifiersEnd - 1);
  for (index[i] = indexRange.getLowerCorner()[i];
       index[i] != indexRange.getUpperCorner()[i] && ! stop;
       ++index[i]) {
    flip = ! flip;
    for (index[j] = begin[flip]; index[j] != end[flip] && ! stop;
         index[j] += stride[flip]) {
      // If this is an element that we are partitioning.
      if ((*identifiers)(index) == identifiersEnd - 1) {
        // If we should put this element in the lower partition.
        if (cost + costs(index) / 2 < splittingCost) {
          // Place the element in the lower partition.
          (*identifiers)(index) = identifiersMiddle - 1;
          cost += costs(index);
        }
        else {
          // Record the splitting index.
          splittingIndex = index[i];
          // Stop partitioning.  Break out of the loops.
          stop = true;
        }
      }
    }
  }
  assert(splittingIndex != std::size_t(indexRange.getUpperCorner()[i]));

  // Recurse.
  const std::ptrdiff_t upper = indexRange.getUpperCorner()[i];
  indexRange.setUpperCoordinate(i, splittingIndex + 1);
  partitionRegularGridWithBspTree(costs, identifiers, cost, indexRange,
                                  identifiersBegin, identifiersMiddle,
                                  predictionThreshhold);
  indexRange.setLowerCoordinate(i, splittingIndex);
  indexRange.setUpperCoordinate(i, upper);
  partitionRegularGridWithBspTree(costs, identifiers, totalCost - cost,
                                  indexRange, identifiersMiddle,
                                  identifiersEnd, predictionThreshhold);
}


#if 0
template<typename _T>
inline
void
partitionRegularGridWithBspTree
(const ads::Array<2, _T>& costs, ads::Array<2, int>* identifiers,
 const _T totalCost, geom::SemiOpenInterval<2, int> indexRange,
 int identifiersBegin,  int identifiersEnd)
{
  // If we are down to a single piece or there are no elements.
  if (identifiersEnd - identifiersBegin <= 1 || indexRange.isEmpty()) {
    // Do nothing.
    return;
  }

  const int identifiersMiddle = (identifiersBegin + identifiersEnd) / 2;
  const _T splittingCost = totalCost * (identifiersMiddle - identifiersBegin) /
                           _T(identifiersEnd - identifiersBegin);

  // See if we can tighten the index range.
  tightenIndexRange(*identifiers, &indexRange, identifiersEnd - 1);
  // If there are one or fewer elements.
  if (indexRange.computeContent() <= 1) {
    return;
  }

  // There are at least two elements to partition.

  // Choose the splitting dimension.
  const ads::FixedArray<2, int> extents = indexRange.getUpperCorner() -
                                          indexRange.getLowerCorner();
  const int i = extents.max_index();
  const int j = (i + 1) % 2;

  // Split.
  int splittingIndex = indexRange.getUpperCorner()[i];
  _T cost = 0;
  bool stop = false;
  ads::FixedArray<2, int> index;
  for (index[i] = indexRange.getLowerCorner()[i];
       index[i] != indexRange.getUpperCorner()[i] && ! stop;
       ++index[i]) {
    for (index[j] = indexRange.getLowerCorner()[j];
         index[j] != indexRange.getUpperCorner()[j] && ! stop;
         ++index[j]) {
      // If this is an element that we are partitioning.
      if ((*identifiers)(index) == identifiersEnd - 1) {
        // If we should put this element in the lower partition.
        if (cost + costs(index) / 2 < splittingCost) {
          // Place the element in the lower partition.
          (*identifiers)(index) = identifiersMiddle - 1;
          cost += costs(index);
        }
        else {
          // Record the splitting index.
          splittingIndex = index[i];
          // Stop partitioning.  Break out of the loops.
          stop = true;
        }
      }
    }
  }
  assert(splittingIndex != indexRange.getUpperCorner()[i]);

  // Recurse.
  const int upper = indexRange.getUpperCorner()[i];
  indexRange.setUpperCoordinate(i, splittingIndex + 1);
  partitionRegularGridWithBspTree(costs, identifiers, cost, indexRange,
                                  identifiersBegin, identifiersMiddle);
  indexRange.setLowerCoordinate(i, splittingIndex);
  indexRange.setUpperCoordinate(i, upper);
  partitionRegularGridWithBspTree(costs, identifiers, totalCost - cost,
                                  indexRange, identifiersMiddle,
                                  identifiersEnd);
}
#endif

template<typename _T>
inline
void
partitionRegularGridWithBspTree
(const container::MultiArrayConstRef<_T, 2>& costs,
 container::MultiArrayRef<std::size_t, 2>* identifiers,
 const std::size_t numberOfPartitions,
 const std::size_t predictionThreshhold)
{
  assert(costs.range() == identifiers->range());
  // Initialize by putting all of the elements in the final partition.
  std::fill(identifiers->begin(), identifiers->end(), numberOfPartitions - 1);
  geom::SemiOpenInterval<2, std::ptrdiff_t>
  indexRange(costs.bases(), costs.bases() +
             ext::convert_array<std::ptrdiff_t>(costs.extents()));
  partitionRegularGridWithBspTree(costs, identifiers,
                                  std::accumulate(costs.begin(), costs.end(),
                                      _T(0)),
                                  indexRange, 0, numberOfPartitions,
                                  predictionThreshhold);
}



//---------------------------------------------------------------------------
// 3-D
//---------------------------------------------------------------------------


// Reduce the index range to tightly contain the specified value.
inline
void
tightenIndexRange(const container::MultiArrayConstRef<std::size_t, 3>&
                  identifiers,
                  geom::SemiOpenInterval<3, std::ptrdiff_t>* indexRange,
                  const std::size_t value)
{
  typedef container::MultiArrayConstRef<std::size_t, 3>::IndexList IndexList;
  IndexList index;
  bool hasValue;

  // For each dimension.
  for (std::size_t d = 0; d != 3; ++d) {
    const std::size_t i = d;
    const std::size_t j = (d + 1) % 3;
    const std::size_t k = (d + 2) % 3;
    // Lower.
    do {
      hasValue = false;
      index[i] = indexRange->getLowerCorner()[i];
      for (index[j] = indexRange->getLowerCorner()[j];
           index[j] != indexRange->getUpperCorner()[j] && ! hasValue;
           ++index[j]) {
        for (index[k] = indexRange->getLowerCorner()[k];
             index[k] != indexRange->getUpperCorner()[k] && ! hasValue;
             ++index[k]) {
          if (identifiers(index) == value) {
            hasValue = true;
          }
        }
      }
      if (! hasValue) {
        indexRange->setLowerCoordinate(i, index[i] + 1);
      }
      if (indexRange->isEmpty()) {
        return;
      }
    }
    while (! hasValue);

    // Upper.
    do {
      hasValue = false;
      index[i] = indexRange->getUpperCorner()[i] - 1;
      for (index[j] = indexRange->getLowerCorner()[j];
           index[j] != indexRange->getUpperCorner()[j] && ! hasValue;
           ++index[j]) {
        for (index[k] = indexRange->getLowerCorner()[k];
             index[k] != indexRange->getUpperCorner()[k] && ! hasValue;
             ++index[k]) {
          if (identifiers(index) == value) {
            hasValue = true;
          }
        }
      }
      if (! hasValue) {
        indexRange->setUpperCoordinate(i, index[i]);
      }
      if (indexRange->isEmpty()) {
        return;
      }
    }
    while (! hasValue);
  }
}


template<typename _T>
inline
void
partitionRegularGridWithBspTree
(const container::MultiArrayConstRef<_T, 3>& costs,
 container::MultiArrayRef<std::size_t, 3>* identifiers,
 const _T totalCost, geom::SemiOpenInterval<3, std::ptrdiff_t> indexRange,
 std::size_t identifiersBegin, std::size_t identifiersEnd)
{
  typedef typename container::MultiArrayConstRef<std::size_t, 3>::SizeList
  SizeList;
  typedef typename container::MultiArrayConstRef<std::size_t, 3>::IndexList
  IndexList;

  // If we are down to a single piece or there are no elements.
  if (identifiersEnd - identifiersBegin <= 1 || indexRange.isEmpty()) {
    // Do nothing.
    return;
  }

  const std::size_t identifiersMiddle = (identifiersBegin + identifiersEnd) / 2;
  const _T splittingCost = totalCost * (identifiersMiddle - identifiersBegin) /
                           _T(identifiersEnd - identifiersBegin);

  // See if we can tighten the index range.
  tightenIndexRange(*identifiers, &indexRange, identifiersEnd - 1);
  // If there are one or fewer elements.
  if (indexRange.computeContent() <= 1) {
    return;
  }

  // There are at least two elements to partition.

  // Choose the splitting dimension.  Assign i the index with the longest
  // extent.
  const SizeList extents =
    ext::convert_array<std::size_t>(indexRange.getUpperCorner() -
                                    indexRange.getLowerCorner());
  const std::size_t i = std::max_element(extents.begin(), extents.end()) -
                        extents.begin();
  const std::size_t j = (i + 1) % 3;
  const std::size_t k = (i + 2) % 3;
#if 0
  // CONTINUE: Does this help?
  // Assign k the index with the shortest extent.
  std::size_t j = (i + 1) % 3;
  std::size_t k = (i + 2) % 3;
  if (extents[j] < extents[k]) {
    std::swap(j, k);
  }
#endif

  // Split.
  std::size_t splittingIndex = indexRange.getUpperCorner()[i];
  _T cost = 0;
  bool stop = false;
  IndexList index;
  for (index[i] = indexRange.getLowerCorner()[i];
       index[i] != indexRange.getUpperCorner()[i] && ! stop;
       ++index[i]) {
    for (index[j] = indexRange.getLowerCorner()[j];
         index[j] != indexRange.getUpperCorner()[j] && ! stop;
         ++index[j]) {
      for (index[k] = indexRange.getLowerCorner()[k];
           index[k] != indexRange.getUpperCorner()[k] && ! stop;
           ++index[k]) {
        // If this is an element that we are partitioning.
        if ((*identifiers)(index) == identifiersEnd - 1) {
          // If we should put this element in the lower partition.
          if (cost + costs(index) / 2 < splittingCost) {
            // Place the element in the lower partition.
            (*identifiers)(index) = identifiersMiddle - 1;
            cost += costs(index);
          }
          else {
            // Record the splitting index.
            splittingIndex = index[i];
            // Stop partitioning.  Break out of the loops.
            stop = true;
          }
        }
      }
    }
  }
  assert(splittingIndex != std::size_t(indexRange.getUpperCorner()[i]));

  // Recurse.
  const std::ptrdiff_t upper = indexRange.getUpperCorner()[i];
  indexRange.setUpperCoordinate(i, splittingIndex + 1);
  partitionRegularGridWithBspTree(costs, identifiers, cost, indexRange,
                                  identifiersBegin, identifiersMiddle);
  indexRange.setLowerCoordinate(i, splittingIndex);
  indexRange.setUpperCoordinate(i, upper);
  partitionRegularGridWithBspTree(costs, identifiers, totalCost - cost,
                                  indexRange, identifiersMiddle,
                                  identifiersEnd);
}



template<typename _T>
inline
void
partitionRegularGridWithBspTree
(const container::MultiArrayConstRef<_T, 3>& costs,
 container::MultiArrayRef<std::size_t, 3>* identifiers,
 const std::size_t numberOfPartitions)
{
  assert(costs.range() == identifiers->range());
  // Initialize by putting all of the elements in the final partition.
  std::fill(identifiers->begin(), identifiers->end(), numberOfPartitions - 1);
  geom::SemiOpenInterval<3, std::ptrdiff_t>
  indexRange(costs.bases(), costs.bases() +
             ext::convert_array<std::ptrdiff_t>(costs.extents()));
  partitionRegularGridWithBspTree(costs, identifiers,
                                  std::accumulate(costs.begin(), costs.end(),
                                      _T(0)),
                                  indexRange, 0, numberOfPartitions);
}

} // namespace concurrent
}
