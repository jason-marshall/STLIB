// -*- C++ -*-

#if !defined(__mpi_sort_tcc__)
#error This file is an implementation detail of sort.
#endif

namespace stlib
{
namespace mpi
{


template<typename _T>
inline
void
_mergeSequentialScan(std::vector<_T> const& a, std::vector<_T> const& b,
                     std::vector<_T>* output)
{
  output->resize(a.size() + b.size());
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t n = 0;
  // Loop while both inputs are non-empty.
  while (i != a.size() && j != b.size()) {
    if (a[i] < b[j]) {
      (*output)[n++] = a[i++];
    }
    else {
      (*output)[n++] = b[j++];
    }
  }
  // Process the remainder of the non-empty vector.
  if (i != a.size()) {
    memcpy(&(*output)[n], &a[i], (a.size() - i) * sizeof(_T));
  }
  if (j != b.size()) {
    memcpy(&(*output)[n], &b[j], (b.size() - j) * sizeof(_T));
  }
}


template<typename _T>
inline
void
_mergeBinarySearch(std::vector<_T> const& input1, std::vector<_T> const& input2,
                   std::vector<_T>* output)
{
  output->resize(input1.size() + input2.size());
  _T const* a = &input1[0];
  _T const* const aEnd = &*input1.end();
  _T const* b = &input2[0];
  _T const* const bEnd = &*input2.end();
  _T* c = &(*output)[0];
  std::size_t size;
  // Loop while both inputs are non-empty.
  while (a != aEnd && b != bEnd) {
    if (*a < *b) {
      size = std::upper_bound(a, aEnd, *b) - a;
      memcpy(c, a, size * sizeof(_T));
      a += size;
    }
    else {
      size = std::upper_bound(b, bEnd, *a) - b;
      memcpy(c, b, size * sizeof(_T));
      b += size;
    }
    c += size;
  }
  // Process the remainder of the non-empty vector.
  if (a != aEnd) {
    memcpy(c, a, (aEnd - a) * sizeof(_T));
  }
  else if (b != bEnd) {
    memcpy(c, b, (bEnd - b) * sizeof(_T));
  }
}


template<typename _T, bool _UseBinarySearch>
inline
void
_mergeSorted(std::vector<_T> const& input, std::vector<_T>* const output,
             MPI_Comm const comm,
             std::integral_constant<bool, _UseBinarySearch> /*dummy*/)
{
  int const commSize = mpi::commSize(comm);
  int const commRank = mpi::commRank(comm);

  // Copy the input.
  *output = input;

  // A power of two that is at least the number of processes.
  int n = 1;
  while (n < commSize) {
    n *= 2;
  }

  std::vector<_T> received;
  std::vector<_T> merged;
  for (; n > 1; n /= 2) {
    // If this process is a receiver.
    if (commRank < n / 2) {
      int const sender = commRank + n / 2;
      // If there is a sender.
      if (sender < commSize) {
        // Receive a vector of sorted elements.
        recv(&received, sender, 0, comm);
        // Merge the two vectors.
        if (_UseBinarySearch) {
          _mergeBinarySearch(*output, received, &merged);
        }
        else {
          _mergeSequentialScan(*output, received, &merged);
        }
        output->swap(merged);
      }
    }
    // If this process is a sender.
    else if (commRank < n) {
      int const receiver = commRank - n / 2;
      // Send the vector.
      send(*output, receiver, 0, comm);
      // Only the root process holds the merged cells.
      output->clear();
      output->shrink_to_fit();
    }
  }
}


template<typename _T>
inline
void
mergeSortedSequentialScan(std::vector<_T> const& input,
                          std::vector<_T>* const output,
                          MPI_Comm const comm)
{
  _mergeSorted(input, output, comm, std::integral_constant<bool, false>());
}


template<typename _T>
inline
void
mergeSortedBinarySearch(std::vector<_T> const& input,
                        std::vector<_T>* const output,
                        MPI_Comm const comm)
{
  _mergeSorted(input, output, comm, std::integral_constant<bool, true>());
}


template<typename _T>
inline
void
_mergeSortedPairs(std::vector<std::pair<_T, std::size_t> > const& a,
                  std::vector<std::pair<_T, std::size_t> > const& b,
                  std::vector<std::pair<_T, std::size_t> >* output)
{
  // Check the trivial cases.
  if (a.empty()) {
    *output = b;
    return;
  }
  if (b.empty()) {
    *output = a;
    return;
  }

  output->clear();
  // Initialize the output so we don't have to check for an empty output in 
  // the following loop.
  output->push_back(std::pair<_T, std::size_t>
                    {std::min(a.front().first, b.front().first), 0});
  std::size_t i = 0;
  std::size_t j = 0;
  // Loop while both inputs are non-empty.
  while (i != a.size() && j != b.size()) {
    if (a[i].first < b[j].first) {
      if (output->back().first == a[i].first) {
        output->back().second += a[i].second;
      }
      else {
        output->push_back(a[i]);
      }
      ++i;
    }
    else {
      if (output->back().first == b[j].first) {
        output->back().second += b[j].second;
      }
      else {
        output->push_back(b[j]);
      }
      ++j;
    }
  }
  // Process the remainder of the non-empty vector.
  if (i != a.size()) {
    // Note that the first value may match the back of the output.
    if (output->back().first == a[i].first) {
      output->back().second += a[i++].second;
    }
    output->insert(output->end(), a.begin() + i, a.end());
  }
  if (j != b.size()) {
    if (output->back().first == b[j].first) {
      output->back().second += b[j++].second;
    }
    output->insert(output->end(), b.begin() + j, b.end());
  }
}


template<typename _T>
inline
void
mergeSorted(std::vector<std::pair<_T, std::size_t> > const& input,
            std::vector<std::pair<_T, std::size_t> >* output,
            MPI_Comm const comm)
{
  typedef std::pair<_T, std::size_t> Pair;

  int const commSize = mpi::commSize(comm);
  int const commRank = mpi::commRank(comm);

  // Copy the input.
  *output = input;

  // A power of two that is at least the number of processes.
  int n = 1;
  while (n < commSize) {
    n *= 2;
  }

  std::vector<Pair> received;
  std::vector<Pair> merged;
  for (; n > 1; n /= 2) {
    // If this process is a receiver.
    if (commRank < n / 2) {
      int const sender = commRank + n / 2;
      // If there is a sender.
      if (sender < commSize) {
        // Receive a vector of sorted elements.
        recv(&received, sender, 0, comm);
        // Merge the two vectors.
        _mergeSortedPairs(*output, received, &merged);
        output->swap(merged);
      }
    }
    // If this process is a sender.
    else if (commRank < n) {
      int const receiver = commRank - n / 2;
      // Send the vector.
      send(*output, receiver, 0, comm);
      // Only the root process holds the merged cells.
      output->clear();
      output->shrink_to_fit();
    }
  }
}


} // namespace mpi
}
