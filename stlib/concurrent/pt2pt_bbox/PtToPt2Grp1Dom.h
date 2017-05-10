// -*- C++ -*-

/*!
  \file PtToPt2Grp1Dom.h
  \brief Point-to-point communication scheme, 2 groups, 1 domain.
*/

#if !defined(__pt2pt_bbox_PtToPt2Grp1Dom_h__)
#define __pt2pt_bbox_PtToPt2Grp1Dom_h__

#include "stlib/geom/kernel/BBox.h"

#include <vector>

#include <mpi.h>

namespace stlib
{
namespace concurrent
{

//! Point-to-point communication scheme, 2 groups, 1 domain.
/*!
  \param N is the dimension.
  \param T is the real number type.  By default it is double.
  \param OurInfo is the information sent to the other group.
  By default it is int.
  \param TheirInfo is the information received from the other group.
  By default it is int.  (Typically the processor ranks are exchanged.)

  This class determines the point-to-point communication pattern for two
  groups of processors.  Each processor has a rectilinear-domain.  Each
  processor communicates with the processors in the other group whose domains
  overlap its own.

  To determine the communication pattern each group first gathers the domains
  to their root processors.  Then the root processors exchange the gathered
  domains.  After broadcasting the other groups' domains, each processor
  can determine which domains intersect its own.

  In addition to the domain, some information is gathered, exchanged
  and broadcasted.  As a minimum, the processor ranks in the world
  communicator must be carried along.  Additional data is application
  dependent.  For example one might pass the message size in the
  following point-to-point communication along with the processor
  ranks.  See the Eulerian/Lagrangian Coupling (ELC) package for an
  example of using this point-to-point communication scheme.
*/
template < std::size_t N, typename T = double,
           typename OurInfo = int, typename TheirInfo = int >
class PtToPt2Grp1Dom
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

  enum {TAG_EXCHANGE_DOMAINS, TAG_EXCHANGE_INFO};

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

  // Our gathered domains.
  std::vector<bbox_type> _our_domains;
  // Their gathered domains.
  std::vector<bbox_type> _their_domains;
  // Our gathered information.
  std::vector<our_info_type> _our_info;
  // Their gathered information.
  std::vector<their_info_type> _their_info;

private:

  //
  // Not implemented.
  //

  // Default constructor not implemented.
  PtToPt2Grp1Dom();

  // Copy constructor not implemented.
  PtToPt2Grp1Dom(const PtToPt2Grp1Dom&);

  // Assignment operator not implemented.
  PtToPt2Grp1Dom&
  operator=(const PtToPt2Grp1Dom&);

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

  //! Construct from this groups' MPI intracommunicator and other groups' size and root.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  PtToPt2Grp1Dom(const MPI::Comm& comm, const MPI::Intracomm& intracomm,
                 const int their_size, const int their_root) :
    _comm(comm.Dup()),
    _intracomm(intracomm.Dup()),
    _their_size(their_size),
    _their_root(their_root),
    _our_domains(_intracomm.Get_size()),
    _their_domains(_their_size),
    _our_info(_intracomm.Get_size()),
    _their_info(_their_size) {}
#else
  PtToPt2Grp1Dom(const MPI_Comm comm, const MPI_Comm intracomm,
                 const int their_size, const int their_root) :
    _comm(),
    _intracomm(),
    _their_size(their_size),
    _their_root(their_root),
    _our_domains(),
    _their_domains(_their_size),
    _our_info(),
    _their_info(_their_size)
  {
    int size;
    MPI_Comm_size(intracomm, &size);
    _our_domains.resize(size);
    _our_info.resize(size);
    MPI_Comm_dup(comm, &_comm);
    MPI_Comm_dup(intracomm, &_intracomm);
  }
#endif

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  //! Destructor.  Free the communicator.
  ~PtToPt2Grp1Dom()
  {
    _comm.Free();
    _intracomm.Free();
  }
#else
  //! Destructor.  Free the communicator.
  ~PtToPt2Grp1Dom()
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
    \param domain is the Cartesian domain of this processor.
    \param info is this processors' information.
    \param overlap_domains is an output iterator for a container which holds
    the domains from processors that intersect \c domain.
    \param overlap_info is an output iterator for a container which holds
    the info from processors with domains that intersect \c domain.
  */
  template<typename DomainOutIter, typename TheirInfoOutIter>
  void
  solve(const bbox_type& domain, const our_info_type& info,
        DomainOutIter overlap_domains, TheirInfoOutIter overlap_info);

  // @}
};

} // namespace concurrent
}

#define __pt2pt_bbox_PtToPt2Grp1Dom_ipp__
#include "stlib/concurrent/pt2pt_bbox/PtToPt2Grp1Dom.ipp"
#undef __pt2pt_bbox_PtToPt2Grp1Dom_ipp__

#endif
