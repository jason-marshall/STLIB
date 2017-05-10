// -*- C++ -*-

#if !defined(__container_SparseVector_ipp__)
#error This file is an implementation detail of the class SparseVector.
#endif

namespace stlib
{
namespace container
{

//
// Constructors etc.
//

// Construct a sparse vector from a dense vector of possibly different value
// type.
template<typename _T>
template<typename _T2>
inline
SparseVector<_T>::
SparseVector(const std::vector<_T2>& array, const _T2& nullValue) :
  _data()
{
  // For each element in the vector.
  for (std::size_t i = 0; i != array.size(); ++i) {
    // If the value is non-null.
    if (array[i] != nullValue) {
      // Record the index and value.
      _data.push_back(value_type(i, array[i]));
    }
  }
}

//
// Accessors.
//

template<typename _T>
inline
typename SparseVector<_T>::const_iterator
SparseVector<_T>::
find(const key_type index) const
{
  const_iterator i = lower_bound(index);
  if (i == end() || i->first != index) {
    return end();
  }
  return i;
}

// CONTINUE
#if 0
template<typename _T>
template<typename T2>
inline
void
SparseVector<_T>::
fill(std::vector<_T2>* array) const
{
  // First set all the elements to the null value.
  *array = getNull();
  // Then fill in the non-null values.
  fillNonNull(array);
}


template<typename _T>
template<typename T2>
inline
void
SparseVector<_T>::
fillNonNull(std::vector<_T2>* array) const
{
  // The range for the dense array.
  const int lb = array->lbound(0);
  const int ub = array->ubound(0);
  // Do a binary search to find the lower bound of the index range.
  // Index iterator.
  Array<1, int>::const_iterator ii =
    std::lower_bound(_indices.begin(), _indices.end(), lb);
  // Initialize the value iterator.
  typename Array<1, value_type>::const_iterator vi = begin() +
      (ii - _indices.begin());
  // Loop over the index range.
  for (; ii != _indices.end() && *ii < ub; ++ii, ++vi) {
    (*array)(*ii) = *vi;
  }
}
#endif

//
// Manipulators.
//

template<typename _T>
inline
typename SparseVector<_T>::iterator
SparseVector<_T>::
find(const key_type index)
{
  iterator i = lower_bound(index);
  if (i == end() || i->first != index) {
    return end();
  }
  return i;
}

template<typename _T>
inline
std::pair<typename SparseVector<_T>::iterator, bool>
SparseVector<_T>::
insert(const value_type& x)
{
  iterator i = std::lower_bound(_data.begin(), _data.end(), x, value_comp());
  // If the element is not already present.
  if (i == end() || i->first != x.first) {
    i = _data.insert(i, x);
  }
  return std::pair<iterator, bool>(i, true);
}

//---------------------------------------------------------------------------
// Mathematical Operations on Two Sparse Vectors.
//---------------------------------------------------------------------------

// Use the binary function to compute the result.
template<typename _T, typename _BinaryFunction>
SparseVector<_T>
computeBinaryOperation(const SparseVector<_T>& x, const SparseVector<_T>& y,
                       const _BinaryFunction& function)
{
  const _T Null = 0;

  SparseVector<_T> result;
  _T f;
  // Loop over the common index range.
  typename SparseVector<_T>::const_iterator i = x.begin(), j = y.begin();
  while (i != x.end() && j != y.end()) {
    // If the first index is less.
    if (i->first < j->first) {
      f = function(i->second, Null);
      if (f != Null) {
        result.append(i->first, f);
      }
      ++i;
    }
    // If the second index is less.
    else if (j->first < i->first) {
      f = function(Null, j->second);
      if (f != Null) {
        result.append(j->first, f);
      }
      ++j;
    }
    // If the indices are equal.
    else {
      f = function(i->second, j->second);
      if (f != Null) {
        result.append(i->first, f);
      }
      ++i;
      ++j;
    }
  }
  // Loop over the remaining indices of x.
  while (i != x.end()) {
    f = function(i->second, Null);
    if (f != Null) {
      result.append(i->first, f);
    }
    ++i;
  }
  // Loop over the remaining indices of y.
  while (j != y.end()) {
    f = function(Null, j->second);
    if (f != Null) {
      result.append(j->first, f);
    }
    ++j;
  }

  return result;
}


}
}

namespace std
{

//---------------------------------------------------------------------------
// Operations with vectors and sparse vectors.
//---------------------------------------------------------------------------

// += on the non-null elements.
template<typename _T1, typename _T2>
inline
vector<_T1>&
operator+=(vector<_T1>& x, const stlib::container::SparseVector<_T2>& y)
{
  for (typename stlib::container::SparseVector<_T2>::const_iterator i = y.begin();
       i != y.end(); ++i) {
    x[i->first] += i->second;
  }
  return x;
}

// -= on the non-null elements.
template<typename _T1, typename _T2>
inline
vector<_T1>&
operator-=(vector<_T1>& x, const stlib::container::SparseVector<_T2>& y)
{
  for (typename stlib::container::SparseVector<_T2>::const_iterator i = y.begin();
       i != y.end(); ++i) {
    x[i->first] -= i->second;
  }
  return x;
}

// *= on the non-null elements.
template<typename _T1, typename _T2>
inline
vector<_T1>&
operator*=(vector<_T1>& x, const stlib::container::SparseVector<_T2>& y)
{
  for (typename stlib::container::SparseVector<_T2>::const_iterator i = y.begin();
       i != y.end(); ++i) {
    x[i->first] *= i->second;
  }
  return x;
}

// /= on the non-null elements.
template<typename _T1, typename _T2>
inline
vector<_T1>&
operator/=(vector<_T1>& x, const stlib::container::SparseVector<_T2>& y)
{
  for (typename stlib::container::SparseVector<_T2>::const_iterator i = y.begin();
       i != y.end(); ++i) {
#ifdef STLIB_DEBUG
    assert(i->second != 0);
#endif
    x[i->first] /= i->second;
  }
  return x;
}

// %= on the non-null elements.
template<typename _T1, typename _T2>
inline
vector<_T1>&
operator%=(vector<_T1>& x, const stlib::container::SparseVector<_T2>& y)
{
  for (typename stlib::container::SparseVector<_T2>::const_iterator i = y.begin();
       i != y.end(); ++i) {
#ifdef STLIB_DEBUG
    assert(i->second != 0);
#endif
    x[i->first] %= i->second;
  }
  return x;
}

// Perform x += a * y on the non-null elements.
template<typename _T1, typename _T2, typename _T3>
inline
void
scaleAdd(vector<_T1>* x, const _T2 a, const stlib::container::SparseVector<_T3>& y)
{
  for (typename stlib::container::SparseVector<_T3>::const_iterator
         i = y.begin(); i != y.end(); ++i) {
    (*x)[i->first] += a * i->second;
  }
}

} // namespace container
