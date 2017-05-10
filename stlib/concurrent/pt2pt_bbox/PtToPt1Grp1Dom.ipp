// -*- C++ -*-

#if !defined(__pt2pt_bbox_PtToPt1Grp1Dom_ipp__)
#error This file is an implementation detail of the class PtToPt1Grp1Dom.
#endif

namespace stlib
{
namespace concurrent
{

//
// Communication
//

template<std::size_t N, typename T, typename Info>
template<typename DomainOutIter, typename InfoOutIter>
inline
void
PtToPt1Grp1Dom<N, T, Info>::
solve(const bbox_type& domain, const info_type& info,
      DomainOutIter overlap_domains, InfoOutIter overlap_info)
{
  //
  // Gather the data domains and information.
  //

  // CONTINUE: Do this with one Allgather.
#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  _intracomm.Allgather(&domain, sizeof(bbox_type), MPI::BYTE,
                       &_domains[0], sizeof(bbox_type), MPI::BYTE);
  _intracomm.Allgather(&info, sizeof(info_type), MPI::BYTE,
                       &_info[0], sizeof(info_type), MPI::BYTE);
#else
  bbox_type d(domain);
  MPI_Allgather(&d, sizeof(bbox_type), MPI_BYTE,
                &_domains[0], sizeof(bbox_type), MPI_BYTE, _intracomm);
  info_type i(info);
  MPI_Allgather(&i, sizeof(info_type), MPI_BYTE,
                &_info[0], sizeof(info_type), MPI_BYTE, _intracomm);
#endif

  //
  // Test each of the domains for intersection with the interest domain.
  //

#ifdef PT2PT_BBOX_USE_CPP_INTERFACE
  const int rank = _intracomm.Get_rank();
#else
  int rank;
  MPI_Comm_rank(_intracomm, &rank);
#endif
  for (std::size_t n = 0; n != _domains.size(); ++n) {
    // If the domains overlap.
    if (n != std::size_t(rank) && doOverlap(domain, _domains[n])) {
      // Add the domain to the overlapping domains.
      *overlap_domains++ = _domains[n];
      // Add their data to the overlap data.
      *overlap_info++ = _info[n];
    }
  }
}

} // namespace concurrent
}
