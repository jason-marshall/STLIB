// -*- C++ -*-

#if !defined(__geom_kernel_orientation_ipp__)
#error This file is an implementation detail of orientation.
#endif

namespace stlib
{
namespace geom
{


template<typename T>
inline
T
computeOrientationDeterminant(const std::array<T, 2>& a,
                              const std::array<T, 2>& b,
                              const std::array<T, 2>& c)
{
  return (b[0] * c[1] + a[0] * b[1] + c[0] * a[1]
          - b[0] * a[1] - a[0] * c[1] - c[0] * b[1]);
}


template<typename T>
inline
T
computeInCircleDeterminant(const std::array<T, 2>& a,
                           const std::array<T, 2>& b,
                           const std::array<T, 2>& c,
                           const std::array<T, 2>& d)
{
  ads::SquareMatrix<4, T> matrix;
  typename ads::SquareMatrix<4, T>::iterator i = matrix.begin();
  // Row 0
  *i++ = 1;
  *i++ = a[0];
  *i++ = a[1];
  *i++ = a[0] * a[0] + a[1] * a[1];
  // Row 1
  *i++ = 1;
  *i++ = b[0];
  *i++ = b[1];
  *i++ = b[0] * b[0] + b[1] * b[1];
  // Row 2
  *i++ = 1;
  *i++ = c[0];
  *i++ = c[1];
  *i++ = c[0] * c[0] + c[1] * c[1];
  // Row 3
  *i++ = 1;
  *i++ = d[0];
  *i++ = d[1];
  *i++ = d[0] * d[0] + d[1] * d[1];

  return ads::computeDeterminant(matrix);
}

} // namespace geom
}
