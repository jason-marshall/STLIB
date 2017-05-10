// -*- C++ -*-

#if !defined(__pt2pt_bbox_PtToPt2Grp2Dom_ipp__)
#error This file is an implementation detail of the class PtToPt2Grp2Dom.
#endif

namespace stlib
{
namespace concurrent
{

//
// Communication
//

template<std::size_t N, typename T, typename OurInfo, typename TheirInfo>
template < typename DataDomOutIter,
           typename IntDomOutIter,
           typename SendInfoOutIter,
           typename RcvInfoOutIter >
inline
void
PtToPt2Grp2Dom<N, T, OurInfo, TheirInfo>::
solve(const bbox_type& data_domain,
      const bbox_type& interest_domain,
      const our_info_type& info,
      IntDomOutIter overlap_interest_domains,
      SendInfoOutIter send_info,
      DataDomOutIter overlap_data_domains,
      RcvInfoOutIter receive_info)
{
  //
  // Gather our domains and information.
  //

  // CONTINUE: Do this with one Gather.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  _intracomm.Gather(&data_domain, sizeof(bbox_type), MPI::BYTE,
                    &_our_data_domains[0], sizeof(bbox_type), MPI::BYTE,
                    0);
  _intracomm.Gather(&interest_domain, sizeof(bbox_type), MPI::BYTE,
                    &_our_interest_domains[0], sizeof(bbox_type),
                    MPI::BYTE, 0);
  _intracomm.Gather(&info, sizeof(our_info_type), MPI::BYTE,
                    &_our_info[0], sizeof(our_info_type), MPI::BYTE, 0);
#else
  bbox_type dd(data_domain);
  MPI_Gather(&dd, sizeof(bbox_type), MPI_BYTE,
             &_our_data_domains[0], sizeof(bbox_type), MPI_BYTE, 0,
             _intracomm);
  bbox_type id(interest_domain);
  MPI_Gather(&id, sizeof(bbox_type), MPI_BYTE,
             &_our_interest_domains[0], sizeof(bbox_type),
             MPI_BYTE, 0, _intracomm);
  our_info_type i(info);
  MPI_Gather(&i, sizeof(our_info_type), MPI_BYTE,
             &_our_info[0], sizeof(our_info_type), MPI_BYTE, 0,
             _intracomm);
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
    // Exchange the data domains.
    _comm.Sendrecv
    (&_our_data_domains[0],
     _our_data_domains.size() * sizeof(bbox_type),
     MPI::BYTE, _their_root, TAG_EXCHANGE_DATA_DOMAINS,
     &_their_data_domains[0], _their_size * sizeof(bbox_type),
     MPI::BYTE, _their_root, TAG_EXCHANGE_DATA_DOMAINS);
    // Exchange the interest domains.
    _comm.Sendrecv
    (&_our_interest_domains[0],
     _our_interest_domains.size() * sizeof(bbox_type),
     MPI::BYTE, _their_root, TAG_EXCHANGE_INTEREST_DOMAINS,
     &_their_interest_domains[0], _their_size * sizeof(bbox_type),
     MPI::BYTE, _their_root, TAG_EXCHANGE_INTEREST_DOMAINS);
    // Exchange the information.
    _comm.Sendrecv
    (&_our_info[0], _our_info.size() * sizeof(our_info_type),
     MPI::BYTE, _their_root, TAG_EXCHANGE_INFO,
     &_their_info[0], _their_size * sizeof(their_info_type),
     MPI::BYTE, _their_root, TAG_EXCHANGE_INFO);
#else
    // Exchange the data domains.
    MPI_Status status;
    MPI_Sendrecv
    (&_our_data_domains[0],
     _our_data_domains.size() * sizeof(bbox_type),
     MPI_BYTE, _their_root, TAG_EXCHANGE_DATA_DOMAINS,
     &_their_data_domains[0], _their_size * sizeof(bbox_type),
     MPI_BYTE, _their_root, TAG_EXCHANGE_DATA_DOMAINS, _comm, &status);
    // Exchange the interest domains.
    MPI_Sendrecv
    (&_our_interest_domains[0],
     _our_interest_domains.size() * sizeof(bbox_type),
     MPI_BYTE, _their_root, TAG_EXCHANGE_INTEREST_DOMAINS,
     &_their_interest_domains[0], _their_size * sizeof(bbox_type),
     MPI_BYTE, _their_root, TAG_EXCHANGE_INTEREST_DOMAINS, _comm, &status);
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
  // Broadcast their data domains.
  _intracomm.Bcast(&_their_data_domains[0],
                   _their_data_domains.size() * sizeof(bbox_type),
                   MPI::BYTE, 0);
  // Broadcast their interest domains.
  _intracomm.Bcast(&_their_interest_domains[0],
                   _their_interest_domains.size() * sizeof(bbox_type),
                   MPI::BYTE, 0);
  // Broadcast their information.
  _intracomm.Bcast(&_their_info[0],
                   _their_info.size() * sizeof(their_info_type),
                   MPI::BYTE, 0);
#else
  // Broadcast their data domains.
  MPI_Bcast(&_their_data_domains[0],
            _their_data_domains.size() * sizeof(bbox_type),
            MPI_BYTE, 0, _intracomm);
  // Broadcast their interest domains.
  MPI_Bcast(&_their_interest_domains[0],
            _their_interest_domains.size() * sizeof(bbox_type),
            MPI_BYTE, 0, _intracomm);
  // Broadcast their information.
  MPI_Bcast(&_their_info[0],
            _their_info.size() * sizeof(their_info_type),
            MPI_BYTE, 0, _intracomm);
#endif
  //
  // Test their interest domains for intersection with our data domain.
  //

  for (std::size_t n = 0; n != _their_interest_domains.size(); ++n) {
    // If the domains overlap.
    if (doOverlap(data_domain, _their_interest_domains[n])) {
      // Add their domain to the overlapping domains.
      *overlap_interest_domains++ = _their_interest_domains[n];
      // Add their information to the send information.
      *send_info++ = _their_info[n];
    }
  }

  //
  // Test their data domains for intersection with our interest domain.
  //

  for (std::size_t n = 0; n != _their_data_domains.size(); ++n) {
    // If the domains overlap.
    if (doOverlap(interest_domain, _their_data_domains[n])) {
      // Add their domain to the overlapping domains.
      *overlap_data_domains++ = _their_data_domains[n];
      // Add their information to the receive information.
      *receive_info++ = _their_info[n];
    }
  }
}

} // namespace concurrent
}
