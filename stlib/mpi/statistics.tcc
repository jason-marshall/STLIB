// -*- C++ -*-

#if !defined(__mpi_statistics_tcc__)
#error This file is an implementation detail of statistics.
#endif

namespace stlib
{
namespace mpi
{


template<typename _T, typename _Float>
inline
void
gatherStatistics(_T const value, _T* sum, _Float* mean, _T* min, _T* max,
                 MPI_Comm const comm, int const root)
{
  // Handle the serial case.
  if (comm == MPI_COMM_NULL) {
    *sum = value;
    *mean = value;
    *min = value;
    *max = value;
    return;
  }
  // Concurrent case.
  std::vector<_T> const values = gather(value, comm);
  if (commRank(comm) == root) {
    *sum = 0;
    *min = value;
    *max = value;
    for (std::size_t i = 0; i != values.size(); ++i) {
      *sum += values[i];
      if (values[i] < *min) {
        *min = values[i];
      }
      if (values[i] > *max) {
        *max = values[i];
      }
    }
    *mean = *sum / commSize(comm);
  }
}


template<typename _T>
inline
void
printStatistics(std::ostream& out, std::string const& name, _T const value,
                MPI_Comm const comm, int const root)
{
  // Handle the serial case.
  if (comm == MPI_COMM_NULL) {
    out << name << " = " << value << '\n';
    return;
  }
  _T sum = 0;
  double mean = 0;
  _T min = 0;
  _T max = 0;
  gatherStatistics(value, &sum, &mean, &min, &max, comm, root);
  if (commRank(comm) == root) {
    if (min == max) {
      out << name << " = " << value << '\n';
    }
    else {
      out << name
          << "\n  sum = " << sum
          << ", mean = " << mean
          << ", min = " << min
          << ", max = " << max << '\n';
    }
  }
}


template<typename _T, std::size_t N, typename _Float>
inline
void
gatherStatistics(std::array<_T, N> const& values, std::array<_T, N>* sums,
                 std::array<_Float, N>* means, std::array<_T, N>* minima,
                 std::array<_T, N>* maxima, MPI_Comm comm, int root)
{
  // Handle the serial case.
  if (comm == MPI_COMM_NULL) {
    for (std::size_t i = 0; i != values.size(); ++i) {
      (*sums)[i] = values[i];
      (*means)[i] = values[i];
      (*minima)[i] = values[i];
      (*maxima)[i] = values[i];
    }
    return;
  }
  // Concurrent case.
  std::vector<std::array<_T, N> > gathered;
  gather(values, &gathered, comm);
  if (commRank(comm) == root) {
    sums->fill(0);
    for (std::size_t j = 0; j != values.size(); ++j) {
      (*minima)[j] = values[j];
      (*maxima)[j] = values[j];
    }
    for (std::size_t i = 0; i != gathered.size(); ++i) {
      for (std::size_t j = 0; j != values.size(); ++j) {
        (*sums)[j] += gathered[i][j];
        if (gathered[i][j] < (*minima)[j]) {
          (*minima)[j] = gathered[i][j];
        }
        if (gathered[i][j] > (*maxima)[j]) {
          (*maxima)[j] = gathered[i][j];
        }
      }
    }
    _Float const inverse = _Float(1) / commSize(comm);
    for (std::size_t j = 0; j != values.size(); ++j) {
      (*means)[j] = (*sums)[j] * inverse;
    }
  }
}


template<typename _T, std::size_t N>
inline
void
printStatistics(std::ostream& out, std::array<std::string, N> const& names,
                std::array<_T, N> const& values, MPI_Comm comm, int root)
{
  // Handle the serial case.
  if (comm == MPI_COMM_NULL) {
    for (std::size_t i = 0; i != values.size(); ++i) {
      out << names[i] << " = " << values[i] << '\n';
    }
    return;
  }
  std::array<_T, N> sums;
  std::array<double, N> means;
  std::array<_T, N> minima;
  std::array<_T, N> maxima;
  gatherStatistics(values, &sums, &means, &minima, &maxima, comm, root);
  if (commRank(comm) == root) {
    for (std::size_t i = 0; i != values.size(); ++i) {
      if (minima[i] == maxima[i]) {
        out << names[i] << " = " << values[i] << '\n';
      }
      else {
        out << names[i]
            << "\n  sum = " << sums[i]
            << ", mean = " << means[i]
            << ", min = " << minima[i]
            << ", max = " << maxima[i] << '\n';
      }
    }
  }
}


template<typename _T, typename _Float>
inline
void
gatherStatistics(std::vector<_T> const& values, _T* const sum,
                 _Float* const mean, _T* const min, _T* const max,
                 MPI_Comm const comm, int const root)
{
  gatherStatistics(values.begin(), values.end(), sum, mean, min, max, comm,
                   root);
}


template<typename _T>
inline
void
printStatistics(std::ostream& out, std::string const& name,
                std::vector<_T> const& values, MPI_Comm const comm,
                int const root)
{
  _T sum = 0;
  double mean = 0;
  _T min = 0;
  _T max = 0;
  gatherStatistics(values.begin(), values.end(), &sum, &mean, &min, &max, comm,
                   root);
  if (comm == MPI_COMM_NULL || commRank(comm) == root) {
    out << name
        << "\n  sum = " << sum
        << ", mean = " << mean
        << ", min = " << min
        << ", max = " << max << '\n';
  }
}


template<typename _ForwardIterator, typename _T, typename _Float>
inline
void
gatherStatistics(_ForwardIterator first, _ForwardIterator last, _T* sum,
                 _Float* mean, _T* min, _T* max, MPI_Comm comm, int root)
{
  // First calculate with the local range.
  *sum = std::accumulate(first, last, _T(0));
  *min = (first == last ? std::numeric_limits<_T>::max() :
          *std::min_element(first, last));
  *max = (first == last ? std::numeric_limits<_T>::lowest() :
          *std::max_element(first, last));

  // Handle the serial case.
  if (comm == MPI_COMM_NULL) {
    *mean = _Float(*sum) / std::distance(first, last);
    return;
  }

  int const rank = commRank(comm);
  *sum = reduce(*sum, MPI_SUM, comm, root);
  unsigned long long const size =
    reduce((unsigned long long)(std::distance(first, last)), MPI_SUM, comm,
           root);
  if (rank == root) {
    *mean = _Float(*sum) / size;
  }
  *min = reduce(*min, MPI_MIN, comm, root);
  *max = reduce(*max, MPI_MAX, comm, root);
}


} // namespace mpi
} // namespace stlib
