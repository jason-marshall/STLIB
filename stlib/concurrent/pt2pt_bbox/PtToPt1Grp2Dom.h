// -*- C++ -*-

/*!
  \file PtToPt1Grp2Dom.h
  \brief Point-to-point communication scheme, 1 group, 2 domains.
*/

#if !defined(__pt2pt_bbox_PtToPt1Grp2Dom_h__)
#define __pt2pt_bbox_PtToPt1Grp2Dom_h__

#include "stlib/geom/kernel/BBox.h"

#include <vector>

#include <mpi.h>

namespace stlib
{
namespace concurrent
{

//! Point-to-point communication scheme, 1 group, 2 domains.
/*!
  \param N is the dimension.
  \param T is the real number type.  By default it is double.
  \param Info is the information exchanged.  By default it is int.
  Typically the processor ranks are exchanged.

  This class determines the point-to-point communication pattern for a
  group of processors.  Each processor has two rectilinear-domains, a
  <em>data domain</em> and an <em>interest domain</em>.  As the names
  suggest, the data domain contains the processors data and the interest
  domain contains the region of interest.  Typically, things in the region of
  interest could affect a processors data, so the interest domain is a
  superset of the data domain.  Each processor sends data to processors
  whose interest domains overlap its data domain.  Each processor
  receives data from processors whose data domains overlap its interest
  domain.

  To determine the communication pattern, the domains and information are
  gathered to each processor.  Then each processor finds the gathered
  interest domains that interesect its data domain and then finds
  the gathered data domains that intersect its interest domain.

  As a minimum, the processor ranks must be exchanged.  Additional
  data is application dependent.  For example, one might pass the
  message size (or an upper bound on the message size) in the
  following point-to-point communication along with the processor
  ranks.
*/
template < std::size_t N, typename T = double, typename Info = int >
class PtToPt1Grp2Dom
{
  //
  // Types.
  //

public:

  //! The number type.
  typedef T number_type;
  //! The information to exchange.
  typedef Info info_type;
  //! A bounding box.
  typedef geom::BBox<N, T> bbox_type;

  //
  // Member data.
  //

private:

  // Our intra-communicator.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  MPI::Intracomm _intracomm;
#else
  MPI_Comm _intracomm;
#endif

  // The gathered data domains.
  std::vector<bbox_type> _data_domains;
  // The gathered interest domains.
  std::vector<bbox_type> _interest_domains;
  // The gathered info.
  std::vector<info_type> _info;

private:

  //
  // Not implemented.
  //

  // Default constructor not implemented.
  PtToPt1Grp2Dom();

  // Copy constructor not implemented.
  PtToPt1Grp2Dom(const PtToPt1Grp2Dom&);

  // Assignment operator not implemented.
  PtToPt1Grp2Dom&
  operator=(const PtToPt1Grp2Dom&);

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  //! Construct from this groups' MPI intracommunicator.
  PtToPt1Grp2Dom(const MPI::Intracomm& intracomm) :
    _intracomm(intracomm.Dup()),
    _data_domains(_intracomm.Get_size()),
    _interest_domains(_intracomm.Get_size()),
    _info(_intracomm.Get_size()) {}
#else
  //! Construct from this groups' MPI intracommunicator.
  PtToPt1Grp2Dom(const MPI_Comm intracomm) :
    _intracomm(),
    _data_domains(),
    _interest_domains(),
    _info()
  {
    int size;
    MPI_Comm_size(intracomm, &size);
    _data_domains.resize(size);
    _interest_domains.resize(size);
    _info.resize(size);
    MPI_Comm_dup(intracomm, &_intracomm);
  }
#endif

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  //! Destructor.  Free the communicator.
  ~PtToPt1Grp2Dom()
  {
    _intracomm.Free();
  }
#else
  //! Destructor.  Free the communicator.
  ~PtToPt1Grp2Dom()
  {
    MPI_Comm_free(&_intracomm);
  }
#endif

  // @}
  //--------------------------------------------------------------------------
  //! \name Communication.
  // @{

  //! Determine the communication pattern.
  /*!
    \param data_domain is the bounding box around the data of this processor.
    \param interest_domain is a bounding box containing the region of
    interest to this processor.
    \param info is this processors' information.
    \param overlap_interest_domains is an output iterator for a container
    which holds the interest domains from processors that intersect
    \c data_domain.
    \param send_info is an output iterator for a container which holds
    the information from processors to which we will send data.
    \param overlap_data_domains is an output iterator for a container
    which holds the data domains from processors that intersect
    \c interest_domain.
    \param receive_info is an output iterator for a container which holds
    the information from processors from which we will receive data.
  */
  template < typename DataDomOutIter,
             typename IntDomOutIter,
             typename SendInfoOutIter,
             typename RcvInfoOutIter >
  void
  solve(const bbox_type& data_domain,
        const bbox_type& interest_domain,
        const info_type& info,
        IntDomOutIter overlap_interest_domains,
        SendInfoOutIter send_info,
        DataDomOutIter overlap_data_domains,
        RcvInfoOutIter receive_info);

  // @}
};

} // namespace concurrent
}

#define __pt2pt_bbox_PtToPt1Grp2Dom_ipp__
#include "stlib/concurrent/pt2pt_bbox/PtToPt1Grp2Dom.ipp"
#undef __pt2pt_bbox_PtToPt1Grp2Dom_ipp__

#endif
