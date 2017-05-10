// -*- C++ -*-

/*!
  \file geom/neighbors/FixedRadiusNeighborSearchWithGroups.h
  \brief A class a fixed-radius neighbor search on points that are grouped.
*/

#if !defined(__geom_FixedRadiusNeighborSearchWithGroups_h__)
#define __geom_FixedRadiusNeighborSearchWithGroups_h__

#include "stlib/geom/neighbors/FixedRadiusNeighborSearch.h"

#include <set>

namespace stlib
{
namespace geom
{

//! Fixed-radius neighbor search on points that are grouped.
/*!
  There are a number of groups of points. Each point is a member of a single
  group. If any point of a group is reported in a neighbor query, then all of
  the points in that group are reported as well, whether they are within the
  specified radius or not.
*/
template<std::size_t N, typename _Location>
class FixedRadiusNeighborSearchWithGroups :
  public FixedRadiusNeighborSearch<N, _Location>
{
  //
  // Types.
  //
private:

  typedef FixedRadiusNeighborSearch<N, _Location> Base;

  //
  // Data
  //
private:

  //! The group index for each record.
  std::vector<std::size_t> _groupIndices;
  //! The records arranged into groups.
  container::StaticArrayOfArrays<std::size_t> _groups;
  //! The records in the ball.
  std::vector<std::size_t> _recordsInBall;
  //! The groups with records in the ball.
  std::set<std::size_t> _groupsInBall;

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{
public:

  //! Construct from a range of records, their group indices, and the search radius.
  /*!
    \pre There must be a non-zero number of records.

    \param first The first record.
    \param last The last record.
    \param groupIndices The beginning of the range of group indices. This
    range specifies the group index for each record.
    \param radius The search radius.

    \c first and \c last must be random access iterators because they will
    be used to compute indices.
    The number of cells will be approximately the number of records.
  */
  template<typename _IndexInputIterator>
  FixedRadiusNeighborSearchWithGroups(typename Base::Record first,
                                      typename Base::Record last,
                                      _IndexInputIterator groupIndices,
                                      typename Base::Float radius);

  // @}
  //--------------------------------------------------------------------------
  //! \name Searching.
  // @{
public:

  //! Find the neighbors of the specified record.
  /*!
    Records that are in the same group is the specified record or are
    in a group that has a member within the search radius are
    reported. Note that the record itself is not reported.
  */
  template<typename _RecordIndexOutputIterator>
  void
  findNeighbors(_RecordIndexOutputIterator records,  std::size_t recordIndex);

  //! Find the neighboring records and groups of the specified record.
  /*!
    The reported groups include the group of the specified record and all
    groups containing records within the search radius. Records that are in
    the same group is the specified record or are in a group that has a member
    within the search radius are reported. Note that the record itself is not
    reported.
  */
  template<typename _RecordIndexOutputIterator,
           typename _GroupIndexOutputIterator>
  void
  findNeighbors(_RecordIndexOutputIterator records,
                _GroupIndexOutputIterator groups, std::size_t recordIndex);

  //! Find the neighboring groups of the specified record.
  /*!
    The reported groups include the group of the specified record and all
    groups containing records within the search radius.
  */
  template<typename _GroupIndexOutputIterator>
  void
  findNeighboringGroups(_GroupIndexOutputIterator groups,
                        std::size_t recordIndex);

private:

  void
  determineGroupsInBall(std::size_t recordIndex);

  template<typename _RecordIndexOutputIterator>
  void
  reportRecords(_RecordIndexOutputIterator records, std::size_t recordIndex);

  // @}
};

} // namespace geom
}

#define __geom_FixedRadiusNeighborSearchWithGroups_ipp__
#include "stlib/geom/neighbors/FixedRadiusNeighborSearchWithGroups.ipp"
#undef __geom_FixedRadiusNeighborSearchWithGroups_ipp__

#endif
