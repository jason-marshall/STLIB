// -*- C++ -*-

#if !defined(__levelSet_contentFromDistance_ipp__)
#error This file is an implementation detail of contentFromDistance.
#endif

namespace stlib
{
namespace levelSet
{


// Return the content (length) of the 1-D manifold in 1-D space.
template<typename _InputIterator, typename _T>
inline
_T
contentFromDistance(std::integral_constant<std::size_t, 1> /*Dimension*/,
                    _InputIterator begin, _InputIterator end, const _T dx)
{
  // The content of a voxel.
  const _T voxel = dx;
  // The radius of the ball that has the same content as a voxel.
  const _T r = 0.5 * dx;
  // The linear function between -r and r is y = a * x + b
  const _T a = - voxel / (2 * r);
  const _T b = 0.5 * voxel;
  _T content = 0;
  _T x;
  while (begin != end) {
    x = *begin++;
    if (x <= - r) {
      content += voxel;
    }
    else if (x < r) {
      content += a * x + b;
    }
    // Else the contributed content is zero.
  }
  return content;
}


// Return the content (area) of the 2-D manifold in 2-D space.
template<typename _InputIterator, typename _T>
inline
_T
contentFromDistance(std::integral_constant<std::size_t, 2> /*Dimension*/,
                    _InputIterator begin, _InputIterator end, const _T dx)
{
  // The content of a voxel.
  const _T voxel = dx * dx;
  // The radius of the ball that has the same content as a voxel.
  // pi r^2 = dx^2
  const _T r = dx / std::sqrt(numerical::Constants<_T>::Pi());
  // The linear function between -r and r is y = a * x + b
  const _T a = - voxel / (2 * r);
  const _T b = 0.5 * voxel;
  _T content = 0;
  _T x;
  while (begin != end) {
    x = *begin++;
    if (x <= - r) {
      content += voxel;
    }
    else if (x < r) {
      content += a * x + b;
    }
    // Else the contributed content is zero.
  }
  return content;
}


// Return the content (volume) of the 3-D manifold in 3-D space.
template<typename _InputIterator, typename _T>
inline
_T
contentFromDistance(std::integral_constant<std::size_t, 3> /*Dimension*/,
                    _InputIterator begin, _InputIterator end, const _T dx)
{
  // The content of a voxel.
  const _T voxel = dx * dx * dx;
  // The radius of the ball that has the same content as a voxel.
  // 4 * pi r^3 / 3 = dx^3
  const _T r = dx * std::pow(3 / (4 * numerical::Constants<_T>::Pi()),
                             _T(1. / 3));
  // The linear function between -r and r is y = a * x + b
  const _T a = - voxel / (2 * r);
  const _T b = 0.5 * voxel;
  _T content = 0;
  _T x;
  while (begin != end) {
    x = *begin++;
    if (x <= - r) {
      content += voxel;
    }
    else if (x < r) {
      content += a * x + b;
    }
    // Else the contributed content is zero.
  }
  return content;
}


// Return the content of the manifold.
// Call the appropriate dimension-specific function.
template<std::size_t _D, typename _InputIterator, typename _T>
inline
_T
contentFromDistance(_InputIterator begin, _InputIterator end, _T dx)
{
  return contentFromDistance(std::integral_constant<std::size_t, _D>(),
                             begin, end, dx);
}


} // namespace levelSet
}
