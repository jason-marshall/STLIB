// -*- C++ -*-

#if !defined(__pt2pt_bbox_PtToPt2Grp1Dom_ipp__)
#error This file is an implementation detail of the class PtToPt2Grp1Dom.
#endif

namespace stlib
{
namespace concurrent
{

//
// Communication
//

template<std::size_t N, typename T, typename OurInfo, typename TheirInfo>
template<typename DomainOutIter, typename TheirInfoOutIter>
inline
void
PtToPt2Grp1Dom<N, T, OurInfo, TheirInfo>::
solve(const bbox_type& domain, const our_info_type& info,
      DomainOutIter overlap_domains, TheirInfoOutIter overlap_info)
{
  //
  // Gather our domains and information.
  //

  // CONTINUE: Do this with one Allgather.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  _intracomm.Gather(&domain, sizeof(bbox_type), MPI::BYTE,
                    &_our_domains[0], sizeof(bbox_type), MPI::BYTE, 0);
  _intracomm.Gather(&info, sizeof(our_info_type), MPI::BYTE,
                    &_our_info[0], sizeof(our_info_type), MPI::BYTE, 0);
#else
  bbox_type d(domain);
  MPI_Gather(&d, sizeof(bbox_type), MPI_BYTE,
             &_our_domains[0], sizeof(bbox_type), MPI_BYTE, 0, _intracomm);
  our_info_type i(info);
  MPI_Gather(&i, sizeof(our_info_type), MPI_BYTE,
             &_our_info[0], sizeof(our_info_type), MPI_BYTE, 0, _intracomm);
#endif

  //
  // Exchange the domains and information.
  //

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  const int rank = _intracomm.Get_rank();
#else
  int rank;
  MPI_Comm_rank(_intracomm, &rank);
#endif

  // If this is the root of our intra-communicator.
  if (rank == 0) {
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
    // Exchange the domains.
    _comm.Sendrecv
    (&_our_domains[0], _our_domains.size() * sizeof(bbox_type),
     MPI::BYTE, _their_root, TAG_EXCHANGE_DOMAINS,
     &_their_domains[0], _their_size * sizeof(bbox_type),
     MPI::BYTE, _their_root, TAG_EXCHANGE_DOMAINS);
    // Exchange the information.
    _comm.Sendrecv
    (&_our_info[0], _our_info.size() * sizeof(our_info_type),
     MPI::BYTE, _their_root, TAG_EXCHANGE_INFO,
     &_their_info[0], _their_size * sizeof(their_info_type),
     MPI::BYTE, _their_root, TAG_EXCHANGE_INFO);
#else
    // Exchange the domains.
    MPI_Status status;
    MPI_Sendrecv
    (&_our_domains[0], _our_domains.size() * sizeof(bbox_type),
     MPI_BYTE, _their_root, TAG_EXCHANGE_DOMAINS,
     &_their_domains[0], _their_size * sizeof(bbox_type),
     MPI_BYTE, _their_root, TAG_EXCHANGE_DOMAINS, _comm, &status);
    // Exchange the information.
    MPI_Sendrecv
    (&_our_info[0], _our_info.size() * sizeof(our_info_type),
     MPI_BYTE, _their_root, TAG_EXCHANGE_INFO,
     &_their_info[0], _their_size * sizeof(their_info_type),
     MPI_BYTE, _their_root, TAG_EXCHANGE_INFO, _comm, &status);
#endif
  }

  //
  // Broadcast their domains and information.
  //

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  // Broadcast the domains.
  _intracomm.Bcast(&_their_domains[0],
                   _their_domains.size() * sizeof(bbox_type),
                   MPI::BYTE, 0);
  // Broadcast the information.
  _intracomm.Bcast(&_their_info[0],
                   _their_info.size() * sizeof(their_info_type),
                   MPI::BYTE, 0);
#else
  // Broadcast the domains.
  MPI_Bcast(&_their_domains[0],
            _their_domains.size() * sizeof(bbox_type),
            MPI_BYTE, 0, _intracomm);
  // Broadcast the information.
  MPI_Bcast(&_their_info[0],
            _their_info.size() * sizeof(their_info_type),
            MPI_BYTE, 0, _intracomm);
#endif

  //
  // Test each of their domains for intersection.
  //

  for (std::size_t n = 0; n != _their_domains.size(); ++n) {
    // If the domains overlap.
    if (doOverlap(domain, _their_domains[n])) {
      // Add their domain to the overlapping domains.
      *overlap_domains++ = _their_domains[n];
      // Add their information to the overlap information.
      *overlap_info++ = _their_info[n];
    }
  }
}

} // namespace concurrent
}
