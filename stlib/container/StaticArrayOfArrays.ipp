// -*- C++ -*-

#if !defined(__container_StaticArrayOfArrays_ipp__)
#error This file is an implementation detail of the class StaticArrayOfArrays.
#endif

namespace stlib
{
namespace container
{


// Construct from a container of containers.
template<typename _T>
template<typename _Container>
inline
StaticArrayOfArrays<_T>::
StaticArrayOfArrays(const _Container& cOfC) :
  _elements(),
  _pointers()
{
  // Count the total number of elements.
  std::size_t numberOfElements = 0;
  for (std::size_t i = 0; i != cOfC.size(); ++i) {
    numberOfElements += cOfC[i].size();
  }
  // Allocate memory.
  _elements.resize(numberOfElements);
  _pointers.resize(cOfC.size() + 1);
  // Build the data structure.
  _pointers[0] = begin();
  iterator iter = begin();
  for (std::size_t i = 0; i != cOfC.size(); ++i) {
    _pointers[i + 1] = _pointers[i] + cOfC[i].size();
    iter = std::copy(cOfC[i].begin(), cOfC[i].end(), iter);
  }
  // Validity check.
  assert(_pointers[_pointers.size() - 1] == end());
  assert(iter == end());
}

template<typename _T>
template<typename SizeForwardIter, typename ValueForwardIter>
inline
StaticArrayOfArrays<_T>::
StaticArrayOfArrays(SizeForwardIter sizesBeginning, SizeForwardIter sizesEnd,
                    ValueForwardIter valuesBeginning,
                    ValueForwardIter valuesEnd) :
  _elements(valuesBeginning, valuesEnd),
  _pointers(std::distance(sizesBeginning, sizesEnd) + 1)
{
  _pointers[0] = begin();
  for (size_type n = 0; n != getNumberOfArrays(); ++n, ++sizesBeginning) {
    _pointers[n + 1] = _pointers[n] + *sizesBeginning;
  }
  assert(_pointers[_pointers.size() - 1] == end());
}


template<typename _T>
template<typename SizeForwardIter>
inline
void
StaticArrayOfArrays<_T>::
rebuild(SizeForwardIter sizesBeginning, SizeForwardIter sizesEnd)
{
  _elements.resize(std::accumulate(sizesBeginning, sizesEnd, 0));
  _pointers.resize(std::distance(sizesBeginning, sizesEnd) + 1);
  _pointers[0] = begin();
  for (std::size_t n = 0; n != getNumberOfArrays(); ++n, ++sizesBeginning) {
    _pointers[n + 1] = _pointers[n] + *sizesBeginning;
  }
  assert(_pointers[_pointers.size() - 1] == end());
}


template<typename _T>
inline
void
StaticArrayOfArrays<_T>::
put(std::ostream& out) const
{
  out << getNumberOfArrays() << " " << size() << '\n';
  for (size_type n = 0; n != getNumberOfArrays(); ++n) {
    out << size(n) << '\n';
    std::copy(begin(n), end(n), std::ostream_iterator<value_type>(out, " "));
    out << '\n';
  }
}

template<typename _T>
inline
void
StaticArrayOfArrays<_T>::
get(std::istream& in)
{
  std::size_t numberOfArrays, numberOfElements, sz;
  in >> numberOfArrays >> numberOfElements;
  _elements.resize(numberOfElements);
  _pointers.resize(numberOfArrays + 1);

  _pointers[0] = begin();
  for (size_type n = 0; n != numberOfArrays; ++n) {
    in >> sz;
    _pointers[n + 1] = _pointers[n] + sz;
    for (size_type m = 0; m != sz; ++m) {
      in >> operator()(n, m);
    }
  }
}

} // namespace container
}
