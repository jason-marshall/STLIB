// -*- C++ -*-

#if !defined(__numerical_equality_ipp__)
#error This file is an implementation detail of equality.
#endif

namespace stlib
{
namespace numerical {

template<typename _T>
inline
bool
_areEqual(const _T x, const _T y, const _T toleranceFactor) {
   const _T absX = std::abs(x);
   const _T absY = std::abs(y);
   const _T d = std::abs(x - y);
   const _T eps = toleranceFactor * std::numeric_limits<_T>::epsilon();
   // Use absolute difference for small numbers.
   if (absX <= eps || absY <= eps) {
      return d <= eps;
   }
   // Otherwise use a relative difference.
   return d <= eps * absX && d <= eps * absY;
}


inline
bool
areEqual(const double x, const double y, const double toleranceFactor) {
   return _areEqual(x, y, toleranceFactor);
}


inline
bool
areEqual(const float x, const float y, const float toleranceFactor) {
   return _areEqual(x, y, toleranceFactor);
}


template<typename _T, std::size_t N>
inline
bool
areEqual(const std::array<_T, N>& x, const std::array<_T, N>& y,
         const _T toleranceFactor) {
   return areSequencesEqual(x.begin(), x.end(), y.begin(), toleranceFactor);
}


template<typename _T>
inline
bool
areEqual(const std::vector<_T>& x, const std::vector<_T>& y,
         const _T toleranceFactor) {
   assert(x.size() == y.size());
   return areSequencesEqual(x.begin(), x.end(), y.begin(), toleranceFactor);
}


template<typename _InputIterator1, typename _InputIterator2>
inline
bool
areSequencesEqual(_InputIterator1 begin1, _InputIterator1 end1,
                  _InputIterator2 begin2,
                  const typename
                  std::iterator_traits<_InputIterator1>::value_type
                  toleranceFactor) {
   while (begin1 != end1) {
      if (! areEqual(*begin1++, *begin2++, toleranceFactor)) {
         return false;
      }
   }
   return true;
}


template<typename _T>
inline
bool
_areEqualAbs(const _T x, const _T y, const _T scale, const _T toleranceFactor) {
   return std::abs(x - y) <= scale * toleranceFactor *
      std::numeric_limits<_T>::epsilon();
}


inline
bool
areEqualAbs(const double x, const double y, const double scale,
         const double toleranceFactor) {
   return _areEqualAbs(x, y, scale, toleranceFactor);
}


inline
bool
areEqualAbs(const float x, const float y, const float scale,
         const float toleranceFactor) {
   return _areEqualAbs(x, y, scale, toleranceFactor);
}


template<typename _T, std::size_t N>
inline
bool
areEqualAbs(const std::array<_T, N>& x, const std::array<_T, N>& y,
            const _T scale, const _T toleranceFactor) {
   return areSequencesEqualAbs(x.begin(), x.end(), y.begin(), scale,
                               toleranceFactor);
}


template<typename _T>
inline
bool
areEqualAbs(const std::vector<_T>& x, const std::vector<_T>& y,
            const _T scale, const _T toleranceFactor) {
   assert(x.size() == y.size());
   return areSequencesEqualAbs(x.begin(), x.end(), y.begin(), scale,
                               toleranceFactor);
}


template<typename _InputIterator1, typename _InputIterator2>
inline
bool
areSequencesEqualAbs(_InputIterator1 begin1, _InputIterator1 end1,
                     _InputIterator2 begin2,
                     const typename
                     std::iterator_traits<_InputIterator1>::value_type
                     scale,
                     const typename
                     std::iterator_traits<_InputIterator1>::value_type
                     toleranceFactor) {
   while (begin1 != end1) {
      if (! areEqualAbs(*begin1++, *begin2++, scale, toleranceFactor)) {
         return false;
      }
   }
   return true;
}


template<typename _T>
inline
bool
_isSmall(const _T x, const _T toleranceFactor) {
   return std::abs(x) <= toleranceFactor * std::numeric_limits<_T>::epsilon();
}

inline
bool
isSmall(const double x, const double toleranceFactor) {
   return _isSmall(x, toleranceFactor);
}


inline
bool
isSmall(const float x, const float toleranceFactor) {
   return _isSmall(x, toleranceFactor);
}


} // namespace numerical
}
