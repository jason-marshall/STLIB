// -*- C++ -*-

/*!
  \file PtToPt1Grp1Dom.h
  \brief Point-to-point communication scheme, 1 group, 1 domain.
*/

#if !defined(__pt2pt_bbox_PtToPt1Grp1Dom_h__)
#define __pt2pt_bbox_PtToPt1Grp1Dom_h__

#include "stlib/geom/kernel/BBox.h"

#include <vector>

#include <mpi.h>

namespace stlib
{
namespace concurrent
{

//! Point-to-point communication scheme, 1 group, 1 domain.
/*!
  \param N is the dimension.
  \param T is the real number type.  By default it is double.
  \param Info is the information exchanged.  By default it is int.
  Typically the processor ranks are exchanged.

  This class determines the point-to-point communication pattern for a
  group of processors.  Each processor has a rectilinear-domain.  Each
  processor communicates with the processors whose domains overlap its own.

  To determine the communication pattern, the domains and information are
  gathered to each processor.  Then each processor finds the domains
  that intersect its own.  This determines both the processors which will
  be sent data and the processors from which data will be received.

  As a minimum, the processor ranks must be exchanged.  Additional
  data is application dependent.  For example, one might pass the
  message size (or an upper bound on the message size) in the
  following point-to-point communication along with the processor
  ranks.
*/
template < std::size_t N, typename T = double, typename Info = int >
class PtToPt1Grp1Dom
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

  // The gathered domains.
  std::vector<bbox_type> _domains;
  // The gathered info.
  std::vector<info_type> _info;

private:

  //
  // Not implemented.
  //

  // Default constructor not implemented.
  PtToPt1Grp1Dom();

  // Copy constructor not implemented.
  PtToPt1Grp1Dom(const PtToPt1Grp1Dom&);

  // Assignment operator not implemented.
  PtToPt1Grp1Dom&
  operator=(const PtToPt1Grp1Dom&);

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  //! Construct from this groups' MPI intracommunicator.
  PtToPt1Grp1Dom(const MPI::Intracomm& intracomm) :
    _intracomm(intracomm.Dup()),
    _domains(_intracomm.Get_size()),
    _info(_intracomm.Get_size()) {}
#else
  //! Construct from this groups' MPI intracommunicator.
  PtToPt1Grp1Dom(const MPI_Comm intracomm) :
    _intracomm(),
    _domains(),
    _info()
  {
    int size;
    MPI_Comm_size(intracomm, &size);
    _domains.resize(size);
    _info.resize(size);
    MPI_Comm_dup(intracomm, &_intracomm);
  }
#endif

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  //! Destructor.  Free the communicator.
  ~PtToPt1Grp1Dom()
  {
    _intracomm.Free();
  }
#else
  //! Destructor.  Free the communicator.
  ~PtToPt1Grp1Dom()
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
    \param domain is the Cartesian domain of this processor.
    \param info is this processors' information.
    \param overlap_domains is an output iterator for a container which holds
    the domains from processors that intersect \c domain.
    \param overlap_info is an output iterator for a container which holds
    the information from processors with domains that intersect \c domain.
  */
  template<typename DomainOutIter, typename InfoOutIter>
  void
  solve(const bbox_type& domain, const info_type& info,
        DomainOutIter overlap_domains, InfoOutIter overlap_info);

  // @}
};

} // namespace concurrent
}

#define __pt2pt_bbox_PtToPt1Grp1Dom_ipp__
#include "stlib/concurrent/pt2pt_bbox/PtToPt1Grp1Dom.ipp"
#undef __pt2pt_bbox_PtToPt1Grp1Dom_ipp__

#endif
