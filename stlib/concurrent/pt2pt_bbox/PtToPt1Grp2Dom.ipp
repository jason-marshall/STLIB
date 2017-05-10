// -*- C++ -*-

#if !defined(__pt2pt_bbox_PtToPt1Grp2Dom_ipp__)
#error This file is an implementation detail of the class PtToPt1Grp2Dom.
#endif

namespace stlib
{
namespace concurrent
{

//
// Communication
//

template<std::size_t N, typename T, typename Info>
template < typename DataDomOutIter,
           typename IntDomOutIter,
           typename SendInfoOutIter,
           typename RcvInfoOutIter >
inline
void
PtToPt1Grp2Dom<N, T, Info>::
solve(const bbox_type& data_domain,
      const bbox_type& interest_domain,
      const info_type& info,
      IntDomOutIter overlap_interest_domains,
      SendInfoOutIter send_info,
      DataDomOutIter overlap_data_domains,
      RcvInfoOutIter receive_info)
{
  //
  // Gather the domains and information.
  //

  // CONTINUE: Do this with one Allgather.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  _intracomm.Allgather(&data_domain, sizeof(bbox_type), MPI::BYTE,
                       &_data_domains[0], sizeof(bbox_type), MPI::BYTE);
  _intracomm.Allgather(&interest_domain, sizeof(bbox_type), MPI::BYTE,
                       &_interest_domains[0], sizeof(bbox_type),
                       MPI::BYTE);
  _intracomm.Allgather(&info, sizeof(info_type), MPI::BYTE,
                       &_info[0], sizeof(info_type), MPI::BYTE);
#else
  bbox_type dd(data_domain);
  MPI_Allgather(&dd, sizeof(bbox_type), MPI_BYTE,
                &_data_domains[0], sizeof(bbox_type), MPI_BYTE,
                _intracomm);
  bbox_type id(interest_domain);
  MPI_Allgather(&id, sizeof(bbox_type), MPI_BYTE,
                &_interest_domains[0], sizeof(bbox_type), MPI_BYTE,
                _intracomm);
  info_type i(info);
  MPI_Allgather(&i, sizeof(info_type), MPI_BYTE,
                &_info[0], sizeof(info_type), MPI_BYTE, _intracomm);
#endif

  //
  // See which interest domains intersect our data domain.
  //

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  const int rank = _intracomm.Get_rank();
#else
  int rank;
  MPI_Comm_rank(_intracomm, &rank);
#endif
  for (std::size_t n = 0; n != _interest_domains.size(); ++n) {
    // If the domains overlap.
    if (n != std::size_t(rank) &&
        doOverlap(data_domain, _interest_domains[n])) {
      // Add the interest domain to the overlapping interest domains.
      *overlap_interest_domains++ = _interest_domains[n];
      // Add their information to the send information.
      *send_info++ = _info[n];
    }
  }

  //
  // See which data domains intersect our interest domain.
  //

  for (std::size_t n = 0; n != _data_domains.size(); ++n) {
    // If the domains overlap.
    if (n != std::size_t(rank) &&
        doOverlap(interest_domain, _data_domains[n])) {
      // Add the data domain to the overlapping data domains.
      *overlap_data_domains++ = _data_domains[n];
      // Add their information to the receive information.
      *receive_info++ = _info[n];
    }
  }
}

} // namespace concurrent
}
