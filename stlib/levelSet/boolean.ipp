// -*- C++ -*-

#if !defined(__levelSet_boolean_ipp__)
#error This file is an implementation detail of boolean.
#endif

namespace stlib
{
namespace levelSet
{


//
// Return true if the two regions are equal.
//


// Return true if the two implicit function values are equal.
template<typename _T>
inline
bool
areFunctionValuesEqual(const _T f, const _T g)
{
  return (f != f && g != g) || f == g;
}


template<typename _InputIterator1, typename _InputIterator2>
inline
bool
areFunctionsEqual(_InputIterator1 begin1, _InputIterator1 end1,
                  _InputIterator2 begin2)
{
  while (begin1 != end1) {
    if (! areFunctionValuesEqual(*begin1++, *begin2++)) {
      return false;
    }
  }
  return true;
}


template<typename _Array>
inline
bool
_areFunctionsEqual(const _Array& f, const _Array& g)
{
  for (std::size_t i = 0; i != f.size(); ++i) {
    if (! areFunctionValuesEqual(f[i], g[i])) {
      return false;
    }
  }
  return true;
}


template<typename _T, std::size_t N>
inline
bool
areFunctionsEqual(const container::SimpleMultiArrayConstRef<_T, N>& f,
                  const container::SimpleMultiArrayConstRef<_T, N>& g)
{
  assert(f.extents() == g.extents());
  return areFunctionsEqual(f.begin(), f.end(), g.begin());
}


template<typename _T, std::size_t _D, std::size_t N>
inline
bool
areFunctionsEqual(const Grid<_T, _D, N>& f, const Grid<_T, _D, N>& g)
{
  typedef typename Grid<_T, _D, N>::VertexPatch VertexPatch;

  assert(f.extents() == g.extents());
  for (std::size_t i = 0; i != f.size(); ++i) {
    const VertexPatch& a = f[i];
    const VertexPatch& b = g[i];
    if (a.isRefined()) {
      if (!b.isRefined() ||
          ! areFunctionsEqual(a.begin(), a.end(), b.begin())) {
        return false;
      }
    }
    else {
      if (b.isRefined() || a.fillValue != b.fillValue) {
        return false;
      }
    }
  }
  return true;
}


//
// Calculate the complement of the region.
//

template<typename _ForwardIterator>
inline
void
complement(_ForwardIterator begin, _ForwardIterator end)
{
  for (; begin != end; ++begin) {
    *begin = - *begin;
  }
}


template<typename _T, std::size_t N>
inline
void
complement(container::SimpleMultiArrayRef<_T, N>* f)
{
  complement(f->begin(), f->end());
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
complement(Grid<_T, _D, N>* f)
{
  typedef typename Grid<_T, _D, N>::VertexPatch VertexPatch;

  for (std::size_t i = 0; i != f->size(); ++i) {
    VertexPatch& a = (*f)[i];
    if (a.isRefined()) {
      complement(a.begin(), a.end());
    }
    else {
      a.fillValue = - a.fillValue;
    }
  }
}


//
// Map a binary operation to two sequences.
//

template<typename _BinaryFunctor, typename _InputIterator1,
         typename _InputIterator2, typename _OutputIterator>
inline
void
mapBinary(_BinaryFunctor f, _InputIterator1 begin1, _InputIterator1 end1,
          _InputIterator2 begin2, _OutputIterator out)
{
  while (begin1 != end1) {
    *out++ = f(*begin1++, *begin2++);
  }
}


template<typename _BinaryFunctor, typename _T, std::size_t _D, std::size_t N>
inline
void
mapBinary(_BinaryFunctor functor, const Grid<_T, _D, N>& f,
          const Grid<_T, _D, N>& g, Grid<_T, _D, N>* result)
{
  typedef typename Grid<_T, _D, N>::VertexPatch VertexPatch;

  assert(f.extents() == g.extents());
  assert(f.extents() == result->extents());
  // The output grid may not be either input grid.
  assert(&f != result);
  assert(&g != result);

  // Determine the refined patches.
  {
    std::vector<std::size_t> patchIndices;
    result->clear();
    for (std::size_t i = 0; i != f.size(); ++i) {
      if (f[i].isRefined() || g[i].isRefined()) {
        patchIndices.push_back(i);
      }
    }
    result->refine(patchIndices);
  }

  for (std::size_t i = 0; i != f.size(); ++i) {
    const VertexPatch& a = f[i];
    const VertexPatch& b = g[i];
    if (a.isRefined()) {
      if (b.isRefined()) {
        mapBinary(functor, a.begin(), a.end(), b.begin(),
                  (*result)[i].begin());
      }
      else {
        mapBinary(functor, a.begin(), a.end(),
                  ads::makeSingleValueIterator(b.fillValue),
                  (*result)[i].begin());
      }
    }
    else {
      if (b.isRefined()) {
        mapBinary(functor, b.begin(), b.end(),
                  ads::makeSingleValueIterator(a.fillValue),
                  (*result)[i].begin());
      }
      else {
        (*result)[i].fillValue = functor(a.fillValue, b.fillValue);
      }
    }
  }

  // Remove unecessary refinement.
  result->coarsen();
}


// For two implicit function values, calculate the value of the union.
// Check the special cases that either f or g are not known, i.e. are NaN.
template<typename _T>
inline
_T
unite(const _T f, const _T g)
{
  // If one or more of the points is unkown.
  if (f != f || g != g) {
    // If the first point has a non-positive distance, the union has
    // a non-positive distance.
    if (f <= 0) {
      return f;
    }
    // Likewise for the second point.
    if (g <= 0) {
      return g;
    }
    // Otherwise the state of the union is unknown.
    return std::numeric_limits<_T>::quiet_NaN();
  }
  return std::min(f, g);
}


//
// Calculate the union of the two regions.
//


template<typename _InputIterator1, typename _InputIterator2,
         typename _OutputIterator>
inline
void
unite(_InputIterator1 begin1, _InputIterator1 end1, _InputIterator2 begin2,
      _OutputIterator out)
{
  while (begin1 != end1) {
    *out++ = unite(*begin1++, *begin2++);
  }
}


template<typename _T, std::size_t N>
inline
void
unite(container::SimpleMultiArrayRef<_T, N>* f,
      const container::SimpleMultiArrayConstRef<_T, N>& g)
{
  assert(f->extents() == g.extents());
  unite(f->begin(), f->end(), g.begin(), f->begin());
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
unite(const Grid<_T, _D, N>& f, const Grid<_T, _D, N>& g,
      Grid<_T, _D, N>* result)
{
  Unite<_T> functor;
  mapBinary(functor, f, g, result);
}


// For two implicit function values, calculate the value of the intersection.
// Check the special cases that either f or g are not known, i.e. are NaN.
template<typename _T>
inline
_T
intersect(const _T f, const _T g)
{
  // If one or more of the points is unkown.
  if (f != f || g != g) {
    // If the first point has a non-negative distance, the intersection has
    // a non-negative distance.
    if (f >= 0) {
      return f;
    }
    // Likewise for the second point.
    if (g >= 0) {
      return g;
    }
    // Otherwise the state of the intersection is unknown.
    return std::numeric_limits<_T>::quiet_NaN();
  }
  return std::max(f, g);
}


//
// Calculate the intersection of the two regions.
//


template<typename _Array1, typename _Array2>
inline
void
_intersect(_Array1* f, const _Array2& g)
{
  for (std::size_t i = 0; i != f->size(); ++i) {
    (*f)[i] = intersect((*f)[i], g[i]);
  }
}


template<typename _T, std::size_t N>
inline
void
intersect(container::SimpleMultiArrayRef<_T, N>* f,
          const container::SimpleMultiArrayConstRef<_T, N>& g)
{
  assert(f->extents() == g.extents());
  _intersect(f, g);
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
intersect(const Grid<_T, _D, N>& f, const Grid<_T, _D, N>& g,
          Grid<_T, _D, N>* result)
{
  Intersect<_T> functor;
  mapBinary(functor, f, g, result);
}


// For two implicit function values, calculate the boolean difference.
// Check the special cases that either f or g are not known, i.e. are NaN.
template<typename _T>
inline
_T
difference(const _T f, const _T g)
{
  return intersect(f, -g);
}


//
// Calculate the difference of the two regions.
//


template<typename _Array1, typename _Array2>
inline
void
_difference(_Array1* f, const _Array2& g)
{
  for (std::size_t i = 0; i != f->size(); ++i) {
    (*f)[i] = difference((*f)[i], g[i]);
  }
}


template<typename _T, std::size_t N>
inline
void
difference(container::SimpleMultiArrayRef<_T, N>* f,
           const container::SimpleMultiArrayConstRef<_T, N>& g)
{
  assert(f->extents() == g.extents());
  _difference(f, g);
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
difference(const Grid<_T, _D, N>& f, const Grid<_T, _D, N>& g,
           Grid<_T, _D, N>* result)
{
  Difference<_T> functor;
  mapBinary(functor, f, g, result);
}


} // namespace levelSet
}
