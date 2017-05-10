// -*- C++ -*-

#if !defined(__container_PackedArrayOfArrays_ipp__)
#error This file is an implementation detail of the class PackedArrayOfArrays.
#endif

namespace stlib
{
namespace container
{


template<typename _Integer>
inline
void
transpose(const PackedArrayOfArrays<_Integer>& input,
          PackedArrayOfArrays<_Integer>* transposed)
{
  // Check the trivial case.
  if (input.empty()) {
    transposed->clear();
    return;
  }
  // Determine the number of columns from the array values, then perform the
  // transpose.
  transpose(input,
            std::size_t(*std::max_element(input.begin(), input.end())) + 1,
            transposed);
}


template<typename _Integer>
inline
void
transpose(const PackedArrayOfArrays<_Integer>& input, const std::size_t numCols,
          PackedArrayOfArrays<_Integer>* transposed)
{
  typedef typename PackedArrayOfArrays<_Integer>::iterator iterator;
  typedef typename PackedArrayOfArrays<_Integer>::const_iterator const_iterator;

  // The row indices in the input will become array values in the transposed
  // result. Check that the integer type has sufficient precision.
  if ((input.numArrays() != 0 ? input.numArrays() - 1 : 0) >
      std::numeric_limits<_Integer>::max()) {
    std::ostringstream stream;
    stream << "Error in transpose() for PackedArrayOfArrays. "
           << "There are " << input.numArrays()
           << " rows in the input. The row indices exceed the maximum value "
           << "for the value type, "
           << std::size_t(std::numeric_limits<_Integer>::max())
           << ". Use an integer type with more digits.";
    throw std::overflow_error(stream.str());
  }

  // Determine the sizes of the rows for the transposed array.
  {
    std::vector<std::size_t> sizes(numCols, 0);
    for (std::size_t i = 0; i != input.size(); ++i) {
#ifdef STLIB_DEBUG
      assert(input[i] < sizes.size());
#endif
      ++sizes[input[i]];
    }
    transposed->rebuild(sizes.begin(), sizes.end());
  }
  // Record starting positions within the rows.
  std::vector<iterator> positions(transposed->numArrays());
  for (std::size_t i = 0; i != positions.size(); ++i) {
    positions[i] = transposed->begin(i);
  }
  // Copy and perform the transpose.
  for (std::size_t i = 0; i != input.numArrays(); ++i) {
    for (const_iterator j = input.begin(i); j != input.end(i); ++j) {
      *positions[*j]++ = i;
    }
  }
}


template<typename _Integer, typename _Data>
inline
void
transpose(const PackedArrayOfArrays<std::pair<_Integer, _Data> >& input,
          PackedArrayOfArrays<std::pair<_Integer, _Data> >* transposed)
{
  // Check the trivial case.
  if (input.empty()) {
    transposed->clear();
    return;
  }
  // Determine the number of columns from the array values.
  _Integer maxValue = 0;
  for (std::size_t i = 0; i != input.size(); ++i) {
    if (input[i].first > maxValue) {
      maxValue = input[i].first;
    }
  }
  // Perform the transpose.
  transpose(input, maxValue + 1, transposed);
}


template<typename _Integer, typename _Data>
inline
void
transpose(const PackedArrayOfArrays<std::pair<_Integer, _Data> >& input,
          const std::size_t numCols,
          PackedArrayOfArrays<std::pair<_Integer, _Data> >* transposed)
{
  typedef typename PackedArrayOfArrays<std::pair<_Integer, _Data> >::iterator
    iterator;
  typedef typename PackedArrayOfArrays<std::pair<_Integer, _Data> >::
    const_iterator const_iterator;

  // The row indices in the input will become column indices in the transposed
  // result. Check that the integer type has sufficient precision.
  if ((input.numArrays() != 0 ? input.numArrays() - 1 : 0) >
      std::numeric_limits<_Integer>::max()) {
    std::ostringstream stream;
    stream << "Error in transpose() for PackedArrayOfArrays. "
           << "There are " << input.numArrays()
           << " rows in the input. The row indices exceed the maximum value "
           << "for the value type, "
           << std::size_t(std::numeric_limits<_Integer>::max())
           << ". Use an integer type with more digits.";
    throw std::overflow_error(stream.str());
  }

  // Determine the sizes of the rows for the transposed array.
  {
    std::vector<std::size_t> sizes(numCols, 0);
    for (std::size_t i = 0; i != input.size(); ++i) {
#ifdef STLIB_DEBUG
      assert(input[i].first < sizes.size());
#endif
      ++sizes[input[i].first];
    }
    transposed->rebuild(sizes.begin(), sizes.end());
  }
  // Record starting positions within the rows.
  std::vector<iterator> positions(transposed->numArrays());
  for (std::size_t i = 0; i != positions.size(); ++i) {
    positions[i] = transposed->begin(i);
  }
  // Copy and perform the transpose.
  for (std::size_t i = 0; i != input.numArrays(); ++i) {
    for (const_iterator j = input.begin(i); j != input.end(i); ++j) {
      *positions[j->first]++ = std::pair<_Integer, _Data>(i, j->second);
    }
  }
}


// Construct from a container of containers.
template<typename _T>
template<typename _Container>
inline
PackedArrayOfArrays<_T>::
PackedArrayOfArrays(const _Container& cOfC) :
  _values(),
  _delimiters(1)
{
  _delimiters[0] = 0;
  typedef typename _Container::const_iterator ArrayIter;
  typedef typename _Container::value_type::const_iterator ElementIter;
  for (ArrayIter i = cOfC.begin(); i != cOfC.end(); ++i) {
    pushArray();
    for (ElementIter j = i->begin(); j != i->end(); ++j) {
      push_back(*j);
    }
  }
}


template<typename _T>
template<typename SizeForwardIter, typename ValueForwardIter>
inline
PackedArrayOfArrays<_T>::
PackedArrayOfArrays(SizeForwardIter sizesBeginning, SizeForwardIter sizesEnd,
                    ValueForwardIter valuesBeginning,
                    ValueForwardIter valuesEnd) :
  _values(valuesBeginning, valuesEnd),
  _delimiters(1)
{
  _delimiters[0] = 0;
  std::partial_sum(sizesBeginning, sizesEnd, std::back_inserter(_delimiters));
}


template<typename _T>
template<typename SizeForwardIter>
inline
void
PackedArrayOfArrays<_T>::
rebuild(SizeForwardIter sizesBeginning, SizeForwardIter sizesEnd)
{
  _delimiters.resize(1);
  _delimiters[0] = 0;
  std::partial_sum(sizesBeginning, sizesEnd, std::back_inserter(_delimiters));
  _values.resize(_delimiters.back());
}


template<typename _T>
inline
void
PackedArrayOfArrays<_T>::
shrink_to_fit()
{
#if (__cplusplus >= 201103L)
  _values.shrink_to_fit();
  _delimiters.shrink_to_fit();
#else
  {
    ValueContainer copy = _values;
    _values.swap(copy);
  }
  {
    std::vector<size_type> copy = _delimiters;
    _delimiters.swap(copy);
  }
#endif
}


template<typename _T>
inline
void
PackedArrayOfArrays<_T>::
append(const PackedArrayOfArrays& other)
{
  // Append the elements.
  _values.insert(end(), other.begin(), other.end());
  // Offset the delimiters.
  std::vector<size_type> d(other._delimiters);
  d += _delimiters.back();
  // Append the delimiters.
  _delimiters.pop_back();
  _delimiters.insert(_delimiters.end(), d.begin(), d.end());
}


template<typename _T>
inline
void
PackedArrayOfArrays<_T>::
rebuild(const std::vector<PackedArrayOfArrays>& parts)
{
  // Handle the trivial case.
  if (parts.empty()) {
    clear();
    return;
  }

  // Determine the number of elements and the number of arrays.
  std::size_t totalValues = 0;
  std::size_t totalArrays = 0;
  for (std::size_t i = 0; i != parts.size(); ++i) {
    totalValues += parts[i].size();
    totalArrays += parts[i].numArrays();
  }

  // Allocate memory for the packed array.
  _values.resize(totalValues);
  _delimiters.resize(totalArrays + 1);
  // Set the guard element for the delimiters.
  _delimiters.back() = totalValues;

  // Positions for the values within the aggregated result.
  std::vector<std::size_t> valuePositions(parts.size());
  valuePositions[0] = 0;
  for (std::size_t i = 1; i != valuePositions.size(); ++i) {
    valuePositions[i] = valuePositions[i - 1] + parts[i - 1].size();
  }
  // Delimiters for the array delimiters within the aggregated result.
  std::vector<std::size_t> arrayDelimiters(parts.size() + 1);
  arrayDelimiters[0] = 0;
  for (std::size_t i = 1; i != arrayDelimiters.size(); ++i) {
    arrayDelimiters[i] = arrayDelimiters[i - 1] + parts[i - 1].numArrays();
  }

  // Copy the values and delimiters from each array.
  #pragma omp parallel for
  for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(parts.size()); ++i) {
    // Copy the array elements.
    std::copy(parts[i].begin(), parts[i].end(), begin() + valuePositions[i]);
    // Copy the delimiters. Note that we leave off the final one.
    std::copy(parts[i]._delimiters.begin(), parts[i]._delimiters.end() - 1,
              _delimiters.begin() + arrayDelimiters[i]);
    // Offset the delimiters to reflect their position within the aggregated
    // result.
    const std::size_t offset = valuePositions[i];
    for (std::size_t j = arrayDelimiters[i]; j != arrayDelimiters[i + 1]; ++j) {
      _delimiters[j] += offset;
    }
  }
}


template<typename _T>
inline
void
PackedArrayOfArrays<_T>::
put(std::ostream& out) const
{
  out << numArrays() << " " << size() << '\n';
  for (size_type n = 0; n != numArrays(); ++n) {
    out << size(n) << '\n';
    std::copy(begin(n), end(n),
              std::ostream_iterator<value_type>(out, " "));
    out << '\n';
  }
}

template<typename _T>
inline
void
PackedArrayOfArrays<_T>::
get(std::istream& in)
{
  std::size_t numberOfArrays, numberOfElements, sz;
  in >> numberOfArrays >> numberOfElements;
  _values.resize(numberOfElements);
  _delimiters.resize(numberOfArrays + 1);

  _delimiters[0] = 0;
  for (size_type n = 0; n != numberOfArrays; ++n) {
    in >> sz;
    _delimiters[n + 1] = _delimiters[n] + sz;
    for (size_type m = 0; m != sz; ++m) {
      in >> operator()(n, m);
    }
  }
}

} // namespace container
}
