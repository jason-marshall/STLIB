// -*- C++ -*-

#if !defined(__geom_FixedRadiusNeighborSearchWithGroups_ipp__)
#error This file is an implementation detail of the class FixedRadiusNeighborSearchWithGroups.
#endif

namespace stlib
{
namespace geom
{

template<std::size_t N, typename _Location>
template<typename _IndexInputIterator>
inline
FixedRadiusNeighborSearchWithGroups<N, _Location>::
FixedRadiusNeighborSearchWithGroups(typename Base::Record first,
                                    typename Base::Record last,
                                    _IndexInputIterator groupIndices,
                                    const typename Base::Float radius) :
  Base(first, last, radius),
  _groupIndices(Base::size()),
  _groups(),
  _recordsInBall(),
  _groupsInBall()
{
  // Copy the group indices.
  for (std::size_t i = 0; i != _groupIndices.size(); ++i) {
    _groupIndices[i] = *groupIndices++;
  }
  //
  // Arrange the records into groups.
  //
  assert(Base::size() != 0);
  // Determine the number of groups.
  std::size_t numberOfGroups = *std::max_element(_groupIndices.begin(),
                               _groupIndices.end()) + 1;
  // Count the number of elements in each group.
  std::vector<std::size_t> groupSizes(numberOfGroups, 0);
  for (std::size_t i = 0; i != _groupIndices.size(); ++i) {
    ++groupSizes[_groupIndices[i]];
  }
  _groups.rebuild(groupSizes.begin(), groupSizes.end());
  // Fill the array of arrays.
  std::fill(groupSizes.begin(), groupSizes.end(), 0);
  for (std::size_t i = 0; i != _groupIndices.size(); ++i) {
    const std::size_t g = _groupIndices[i];
    _groups(g, groupSizes[g]++) = i;
  }
}

//
// Mathematical member functions
//

template<std::size_t N, typename _Location>
template<typename _RecordIndexOutputIterator>
inline
void
FixedRadiusNeighborSearchWithGroups<N, _Location>::
findNeighbors(_RecordIndexOutputIterator records,
              const std::size_t recordIndex)
{
  // Determine the groups in the ball.
  determineGroupsInBall(recordIndex);
  // Report the records.
  reportRecords(records, recordIndex);
}

template<std::size_t N, typename _Location>
template<typename _RecordIndexOutputIterator,
         typename _GroupIndexOutputIterator>
inline
void
FixedRadiusNeighborSearchWithGroups<N, _Location>::
findNeighbors(_RecordIndexOutputIterator records,
              _GroupIndexOutputIterator groups, const std::size_t recordIndex)
{
  // Report the groups.
  findNeighboringGroups(groups, recordIndex);
  // Report the records.
  reportRecords(records, recordIndex);
}

template<std::size_t N, typename _Location>
template<typename _GroupIndexOutputIterator>
inline
void
FixedRadiusNeighborSearchWithGroups<N, _Location>::
findNeighboringGroups(_GroupIndexOutputIterator groups,
                      const std::size_t recordIndex)
{
  // Determine the groups in the ball.
  determineGroupsInBall(recordIndex);
  // Report the groups.
  for (std::set<std::size_t>::const_iterator i = _groupsInBall.begin();
       i != _groupsInBall.end(); ++i) {
    *groups++ = *i;
  }
}


template<std::size_t N, typename _Location>
inline
void
FixedRadiusNeighborSearchWithGroups<N, _Location>::
determineGroupsInBall(const std::size_t recordIndex)
{
  // Get the records that are in the ball.
  _recordsInBall.clear();
  Base::findNeighbors(std::back_inserter(_recordsInBall), recordIndex);
  _groupsInBall.clear();
  // The group of the specified record.
  _groupsInBall.insert(_groupIndices[recordIndex]);
  // The group containing records within the search radius.
  for (std::size_t i = 0; i != _recordsInBall.size(); ++i) {
    _groupsInBall.insert(_groupIndices[_recordsInBall[i]]);
  }
}

template<std::size_t N, typename _Location>
template<typename _RecordIndexOutputIterator>
inline
void
FixedRadiusNeighborSearchWithGroups<N, _Location>::
reportRecords(_RecordIndexOutputIterator records,
              const std::size_t recordIndex)
{
  typedef container::StaticArrayOfArrays<std::size_t>::const_iterator
  const_iterator;
#ifdef STLIB_DEBUG
  assert(! _groupsInBall.empty());
#endif
  // Report the records.
  const_iterator begin, end;
  for (std::set<std::size_t>::const_iterator i = _groupsInBall.begin();
       i != _groupsInBall.end(); ++i) {
    end = _groups.end(*i);
    for (begin = _groups.begin(*i); begin != end; ++begin) {
      if (*begin != recordIndex) {
        *records++ = *begin;
      }
    }
  }

}

} // namespace geom
}
