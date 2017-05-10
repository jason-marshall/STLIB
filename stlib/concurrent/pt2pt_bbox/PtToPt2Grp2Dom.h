// -*- C++ -*-

/*!
  \file PtToPt2Grp2Dom.h
  \brief Point-to-point communication scheme, 2 groups, 2 domains.
*/

#if !defined(__pt2pt_bbox_PtToPt2Grp2Dom_h__)
#define __pt2pt_bbox_PtToPt2Grp2Dom_h__

#include "stlib/geom/kernel/BBox.h"

#include <vector>

#include <mpi.h>

namespace stlib
{
namespace concurrent
{

//! Point-to-point communication scheme, 2 groups, 2 domains.
/*!
  \param N is the dimension.
  \param T is the real number type.  By default it is double.
  \param OurInfo is the information sent to the other group.
  By default it is int.
  \param TheirInfo is the information received from the other group.
  By default it is int.  (Typically the processor ranks are exchanged.)

  This class determines the point-to-point communication pattern for two
  groups of processors.  Each processor has two rectilinear-domains, a
  <em>data domain</em> and an <em>interest domain</em>.  As the names
  suggest, the data domain contains the processors data and the interest
  domain contains the region of interest.  Typically, things in the region of
  interest could affect a processors data, so the interest domain is a
  superset of the data domain.  Each processor sends data to processors
  in the other group whose interest domains overlap its data domain.
  Each processor receives data from processors in the other group whose
  data domains overlap its interest domain.

  To determine the communication pattern each group first gathers the domains
  to their root processors.  Then the root processors exchange the gathered
  domains.  After broadcasting the other groups' domains, each processor
  can determine which domains intersect its own.

  In addition to the domain, some information is gathered, exchanged
  and broadcasted.  As a minimum, the processor ranks in the world
  communicator must be carried along.  Additional data is application
  dependent.
*/
template < std::size_t N, typename T = double,
           typename OurInfo = int, typename TheirInfo = int >
class PtToPt2Grp2Dom
{
  //
  // Types.
  //

public:

  //! The number type.
  typedef T number_type;
  //! This groups' data type.
  typedef OurInfo our_info_type;
  //! The other groups' data type.
  typedef TheirInfo their_info_type;
  //! A bounding box.
  typedef geom::BBox<N, T> bbox_type;

  //
  // Enumerations.
  //

private:

  enum {TAG_EXCHANGE_DATA_DOMAINS, TAG_EXCHANGE_INTEREST_DOMAINS,
        TAG_EXCHANGE_INFO
       };

  //
  // Member data.
  //

private:

  // The communicator that contains both groups.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  MPI::Comm _comm;
#else
  MPI_Comm _comm;
#endif

  // Our intra-communicator.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  MPI::Intracomm _intracomm;
#else
  MPI_Comm _intracomm;
#endif

  // The other groups' size;
  int _their_size;
  // The other groups' root.
  int _their_root;

  // Our gathered data domains.
  std::vector<bbox_type> _our_data_domains;
  // Their gathered data domains.
  std::vector<bbox_type> _their_data_domains;

  // Our gathered interest domains.
  std::vector<bbox_type> _our_interest_domains;
  // Their gathered interest domains.
  std::vector<bbox_type> _their_interest_domains;

  // Our gathered information.
  std::vector<our_info_type> _our_info;
  // Their gathered information.
  std::vector<their_info_type> _their_info;

private:

  //
  // Not implemented.
  //

  // Default constructor not implemented.
  PtToPt2Grp2Dom();

  // Copy constructor not implemented.
  PtToPt2Grp2Dom(const PtToPt2Grp2Dom&);

  // Assignment operator not implemented.
  PtToPt2Grp2Dom&
  operator=(const PtToPt2Grp2Dom&);

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Construct from this groups' MPI intracommunicator and other groups' size and root.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  PtToPt2Grp2Dom(const MPI::Comm& comm, const MPI::Intracomm& intracomm,
                 const int their_size, const int their_root) :
    _comm(comm.Dup()),
    _intracomm(intracomm.Dup()),
    _their_size(their_size),
    _their_root(their_root),
    _our_data_domains(_intracomm.Get_size()),
    _their_data_domains(_their_size),
    _our_interest_domains(_intracomm.Get_size()),
    _their_interest_domains(_their_size),
    _our_info(_intracomm.Get_size()),
    _their_info(_their_size) {}
#else
  PtToPt2Grp2Dom(const MPI_Comm comm, const MPI_Comm intracomm,
                 const int their_size, const int their_root) :
    _comm(),
    _intracomm(),
    _their_size(their_size),
    _their_root(their_root),
    _our_data_domains(),
    _their_data_domains(_their_size),
    _our_interest_domains(),
    _their_interest_domains(_their_size),
    _our_info(),
    _their_info(_their_size)
  {
    int size;
    MPI_Comm_size(intracomm, &size);
    _our_data_domains.resize(size);
    _our_interest_domains.resize(size);
    _our_info.resize(size);
    MPI_Comm_dup(comm, &_comm);
    MPI_Comm_dup(intracomm, &_intracomm);
  }
#endif

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  //! Destructor.  Free the communicator.
  ~PtToPt2Grp2Dom()
  {
    _comm.Free();
    _intracomm.Free();
  }
#else
  //! Destructor.  Free the communicator.
  ~PtToPt2Grp2Dom()
  {
    MPI_Comm_free(&_comm);
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
        const our_info_type& info,
        IntDomOutIter overlap_interest_domains,
        SendInfoOutIter send_info,
        DataDomOutIter overlap_data_domains,
        RcvInfoOutIter receive_info);

  // @}
};

} // namespace concurrent
}

#define __pt2pt_bbox_PtToPt2Grp2Dom_ipp__
#include "stlib/concurrent/pt2pt_bbox/PtToPt2Grp2Dom.ipp"
#undef __pt2pt_bbox_PtToPt2Grp2Dom_ipp__

#endif
