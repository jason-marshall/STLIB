// -*- C++ -*-

#if !defined(__ads_array_ArrayWithNullHoles_ipp__)
#error This file is an implementation detail of the class ArrayWithNullHoles.
#endif

namespace stlib
{
namespace ads
{


// Return true if the element is null.
template<typename T>
inline
bool
ArrayWithNullHoles<T>::
isNull(const std::size_t index) const
{
  // The index should be in the correct range.
  assert(index < size());
  if (_data[index] == _null) {
    return true;
  }
  return false;
}


// Return true if the element is non-null.
template<typename T>
inline
bool
ArrayWithNullHoles<T>::
isNonNull(const std::size_t index) const
{
  return ! isNull(index);
}


// Return the specified element.
// The element may be non-null or null.
template<typename T>
inline
typename ArrayWithNullHoles<T>::ParameterType
ArrayWithNullHoles<T>::
get(const std::size_t index) const
{
  // The index should be in the correct range.
  assert(index < size());
  return _data[index];
}


// Insert an element into the first available hole (or at the end if there
// are none).  Return the index of the element.
template<typename T>
inline
std::size_t
ArrayWithNullHoles<T>::
insert(ParameterType value)
{
  // If there are no holes.
  if (_holes.empty()) {
    _data.push_back(value);
    return _data.size() - 1;
  }
  // Otherwise, place the element in a hole.
  std::size_t hole = _holes.back();
  _holes.pop_back();
  _data[hole] = value;
  return hole;
}


// Erase the specified element.
// The location must not be a hole.
template<typename T>
inline
void
ArrayWithNullHoles<T>::
erase(const std::size_t index)
{
  assert(isNonNull(index));
  _holes.push_back(index);
  _data[index] = _null;
}


// Erase a range of elements.
// The location of each must not already be a hole.
template<typename T>
template<typename IntInputIterator>
inline
void
ArrayWithNullHoles<T>::
erase(IntInputIterator begin, IntInputIterator end)
{
  for (; begin != end; ++begin) {
    erase(*begin);
  }
}


// Set the specified element.
template<typename T>
inline
void
ArrayWithNullHoles<T>::
set(const std::size_t index, ParameterType value)
{
  assert(index < size() && value != _null);
  _data[index] = value;
}


// Return true if the data structure is valid.
template<typename T>
inline
bool
ArrayWithNullHoles<T>::
isValid() const
{
  // For each hole.
  for (typename IndexContainer::const_iterator i = _holes.begin();
       i != _holes.end(); ++i) {
    // The element should have a null value.
    if (_data[*i] != _null) {
      return false;
    }
  }

  // Make a set of the holes.
  std::set<std::size_t> holesSet(_holes.begin(), _holes.end());
  // For each element.
  for (std::size_t i = 0; i != _data.size(); ++i) {
    // If the element is null, it must be a hole.
    if (_data[i] == _null && holesSet.count(i) == 0) {
      return false;
    }
    // If the element is non-null, it must not be a hole.
    if (_data[i] != _null && holesSet.count(i) == 1) {
      return false;
    }
  }
  return true;
}

} // namespace ads
} // namespace stlib
