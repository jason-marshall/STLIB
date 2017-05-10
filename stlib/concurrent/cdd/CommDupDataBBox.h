// -*- C++ -*-

/*!
  \file concurrent/cdd/CommDupDataBBox.h
  \brief Communicator for Duplicated Data.
*/

#if !defined(__concurrent_cdd_CommDupDataBBox_h__)
#define __concurrent_cdd_CommDupDataBBox_h__

#include "stlib/concurrent/pt2pt_bbox/PtToPt1Grp2Dom.h"

#include "stlib/ads/iterator/TrivialOutputIterator.h"

#include <mpi.h>

#include <set>

namespace stlib
{
namespace concurrent
{

//! Communicator for Duplicated Data.
/*!
  \param N is the dimension.
  \param Datum is the datum type.  By by default it is \c double.
  \param T is the number type.  By default it is \c double.
  \param ID is the identifier type.  By default it is \c int.

  CONTINUE.
*/
template < int N, typename Datum = double, typename T = double,
           typename ID = int >
class CommDupDataBBox
{
public:

  //
  // Public types.
  //

  //! The data type.
  typedef Datum datum_type;
  //! The number type.
  typedef T number_type;
  //! The identifier type.
  typedef ID id_type;

  //! A point in N-D space.
  typedef ads::FixedArray<N, number_type> point_type;
  //! A bounding box.
  typedef geom::BBox<N, number_type> bbox_type;

private:

  //
  // Enumerations.
  //

  enum {TAG_IDENTIFIERS,
        TAG_POSITIONS,
        TAG_SIZE,
        TAG_DATA
       };

private:

  //
  // Member data.
  //

  // The MPI intra-communicator.
  MPI::Intracomm _comm;
  // Determines the point-to-point communication pattern.
  concurrent::PtToPt1Grp2Dom<N, number_type> _pt2pt;

  // The data identifiers.
  ads::Array<1, const id_type, false> _identifiers;
  // The data positions.
  ads::Array<1, const point_type, false> _positions;

  // The data domain.
  bbox_type _data_domain;
  // The interest domain.
  bbox_type _interest_domain;

  // The set of identifiers.
  std::set<id_type> _identifier_set;

  // The overlapping interest domains.
  std::vector<bbox_type> _overlap_interest_domains;
  // The processors to which we will send data.
  std::vector<int> _send_ranks;
  // The processors from which we will receive data.
  std::vector<int> _receive_ranks;

private:

  //
  // Not implemented.
  //

  // Default constructor not implemented.
  CommDupDataBBox();

  // Copy constructor not implemented.
  CommDupDataBBox(const CommDupDataBBox&);

  // Assignment operator not implemented.
  CommDupDataBBox&
  operator=(const CommDupDataBBox&);

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Construct from the intra-communicator.
  CommDupDataBBox(MPI::Intracomm& comm) :
    _comm(comm),
    _pt2pt(_comm),
    _identifiers(),
    _positions(),
    _data_domain(),
    _interest_domain(),
    _identifier_set(),
    _overlap_interest_domains(),
    _send_ranks(),
    _receive_ranks() {}

  //! Destructor.
  virtual
  ~CommDupDataBBox() {}

  // @}
  //--------------------------------------------------------------------------
  //! \name Communication.
  // @{

  //! Determine the communication pattern.
  void
  determine_communication_pattern();

  //! Exchange the data.
  template <typename IDOutputIterator, typename DataOutputIterator>
  void
  exchange(const datum_type* data, IDOutputIterator received_ids,
           DataOutputIterator received_data);

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{

  //! Set the data identifiers.
  void
  set_identifiers(const int size, const id_type* identifiers)
  {
    // Wrap the array of identifiers.
    _identifiers = ads::Array<1, const id_type, false>(size, identifiers);
    // Build the set of identifiers.
    _identifier_set.clear();
    _identifier_set.insert(_identifiers.begin(), _identifiers.end());
  }

  //! Set the data positions.
  void
  set_positions(const int size, const void* positions)
  {
    _positions = ads::Array<1, const point_type, false>(size, positions);
  }

  //! Set the data domain.
  /*!
    We will send data to any processor whose interest domain overlaps our
    data domain.
   */
  void
  compute_data_domain()
  {
    _data_domain.bound(_positions.begin(), _positions.end());
  }

  //! Set the interest domain.
  /*!
    We will receive data from any processor whose data domain overlaps our
    interest domain.
   */
  void
  set_interest_domain(const bbox_type& interest_domain)
  {
    _interest_domain = interest_domain;
  }

  //! Compute the interest domain from the data domain and an offset.
  /*!
    We will receive data from any processor whose data domain overlaps our
    interest domain.
   */
  void
  compute_interest_domain(const number_type offset)
  {
    _interest_domain.setLowerCorner(_data_domain.getLowerCorner() - offset);
    _interest_domain.setUpperCorner(_data_domain.getUpperCorner() + offset);
  }

  // @}
};

} // namespace concurrent
}

#define __cdd_CommDupDataBBox_ipp__
#include "stlib/concurrent/cdd/CommDupDataBBox.ipp"
#undef __cdd_CommDupDataBBox_ipp__

#endif
