// -*- C++ -*-

#if !defined(__geom_content_ipp__)
#error This file is an implementation detail of content.
#endif

#include <cassert>
#include <cmath>

namespace stlib
{
namespace geom
{


//
// Distance.
//


// Return the distance between the points x and y.
// Specialization for 1-D: signed distance.
template<typename T>
inline
T
computeDistance(const std::array<T, 1>& x, const std::array<T, 1>& y)
{
  return y[0] - x[0];
}


// Return the distance between the points x and y.
template<std::size_t N, typename T>
inline
T
computeDistance(const std::array<T, N>& x, const std::array<T, N>& y)
{
  return ext::euclideanDistance(x, y);
}

// Return the distance between the points x[0] and x[1].
template<std::size_t N, typename T>
inline
T
computeDistance(const std::array<std::array<T, N>, 2>& x)
{
  return computeDistance(x[0], x[1]);
}


// Return the distance between the points x and y.
template<std::size_t N, typename T>
inline
T
computeContent(const std::array<T, N>& x, const std::array<T, N>& y)
{
  return computeDistance(x, y);
}


// Return the distance between the points x[0] and x[1].
template<std::size_t N, typename T>
inline
T
computeContent(const std::array<std::array<T, N>, 2>& x)
{
  return computeDistance(x[0], x[1]);
}


//
// Gradient of distance.
//


// Calculate the gradient with respect to x of the distance between
// the points x and y.
// Specialization for 1-D.
template<typename T>
inline
void
computeGradientOfDistance(const std::array<T, 1>& /*x*/,
                          const std::array<T, 1>& /*y*/,
                          std::array<T, 1>* gradient)
{
  (*gradient)[0] = -1;
}


// Return gradient with respect to x of the distance between the
// points x and y.
template<std::size_t N, typename T>
inline
void
computeGradientOfDistance(const std::array<T, N>& x,
                          const std::array<T, N>& y,
                          std::array<T, N>* gradient)
{
  const T d = computeDistance(x, y);
  for (std::size_t n = 0; n != N; ++n) {
    (*gradient)[n] = (x[n] - y[n]) / d;
  }
}


// Calculate the gradient (with respect to x[0]) of the distance between
// the points \c x[0] and \c x[1].
template<std::size_t N, typename T>
inline
void
computeGradientOfDistance(const std::array<std::array<T, N>, 2>& x,
                          std::array<T, N>* gradient)
{
  computeGradientOfDistance(x[0], x[1], gradient);
}


// Calculate the gradient (with respect to x) of the distance between
// the points \c x and \c y.
template<std::size_t N, typename T>
inline
void
computeGradientOfContent(const std::array<T, N>& x,
                         const std::array<T, N>& y,
                         std::array<T, N>* gradient)
{
  computeGradientOfDistance(x, y, gradient);
}


// Calculate the gradient (with respect to x[0]) of the distance between
// the points \c x[0] and \c x[1].
template<std::size_t N, typename T>
inline
void
computeGradientOfContent(const std::array<std::array<T, N>, 2>& x,
                         std::array<T, N>* gradient)
{
  computeGradientOfDistance(x, gradient);
}




//
// Area.
//

// Return the signed area of the triangle with 2-D points a, b and c.
template<typename T>
inline
T
computeArea(const std::array<T, 2>& a, const std::array<T, 2>& b,
            const std::array<T, 2>& c)
{
  return (a[0] * b[1] + b[0] * c[1] + c[0] * a[1] -
          a[0] * c[1] - b[0] * a[1] - c[0] * b[1]) / 2;
}

// Return the squared area of the triangle with 2-D points a, b and c.
template<typename T>
inline
T
computeSquaredArea(const std::array<T, 2>& a, const std::array<T, 2>& b,
                   const std::array<T, 2>& c)
{
  const T area = computeArea(a, b, c);
  return area * area;
}

// Return the signed area of the triangle with 2-D points p[0], p[1] and p[2].
template<typename T>
inline
T
computeArea(const std::array<std::array<T, 2>, 3>& p)
{
  return computeArea(p[0], p[1], p[2]);
}


// Return the squared area of the triangle with 2-D points p[0], p[1] and p[2].
template<typename T>
inline
T
computeSquaredArea(const std::array<std::array<T, 2>, 3>& p)
{
  return computeSquaredArea(p[0], p[1], p[2]);
}


// Return the unsigned area of the triangle with 3-D points a, b and c.
template<typename T>
inline
T
computeArea(const std::array<T, 3>& a, const std::array<T, 3>& b,
            const std::array<T, 3>& c)
{
  return std::sqrt(computeSquaredArea(a, b, c));
}


// Return the squared area of the triangle with 3-D points a, b and c.
template<typename T>
inline
T
computeSquaredArea(const std::array<T, 3>& a, const std::array<T, 3>& b,
                   const std::array<T, 3>& c)
{
  return T(0.25) * (std::pow(a[0] * b[1] + b[0] * c[1] + c[0] * a[1] -
                             a[0] * c[1] - b[0] * a[1] - c[0] * b[1], 2) +
                    std::pow(a[1] * b[2] + b[1] * c[2] + c[1] * a[2] -
                             a[1] * c[2] - b[1] * a[2] - c[1] * b[2], 2) +
                    std::pow(a[2] * b[0] + b[2] * c[0] + c[2] * a[0] -
                             a[2] * c[0] - b[2] * a[0] - c[2] * b[0], 2));
}


// Return the unsigned area of the triangle with 3-D points p[0], p[1] and p[2].
template<typename T>
inline
T
computeArea(const std::array<std::array<T, 3>, 3>& p)
{
  return computeArea(p[0], p[1], p[2]);
}


// Return the squared area of the triangle with 3-D points p[0], p[1] and p[2].
template<typename T>
inline
T
computeSquaredArea(const std::array<std::array<T, 3>, 3>& p)
{
  return computeSquaredArea(p[0], p[1], p[2]);
}



//
// Gradient of area.
//


// Calculate the gradient (with respect to a) of the signed area
// of the triangle with 2-D points a, b and c.
template<typename T>
inline
void
computeGradientOfArea(const std::array<T, 2>& /*a*/,
                      const std::array<T, 2>& b,
                      const std::array<T, 2>& c,
                      std::array<T, 2>* gradient)
{
  (*gradient)[0] = (b[1] - c[1]) / 2;
  (*gradient)[1] = (c[0] - b[0]) / 2;
}


// Calculate the gradient (with respect to a) of the unsigned area
// of the triangle with 3-D points a, b and c.
template<typename T>
inline
void
computeGradientOfArea(const std::array<T, 3>& a,
                      const std::array<T, 3>& b,
                      const std::array<T, 3>& c,
                      std::array<T, 3>* gradient)
{
  const T ar = computeArea(a, b, c);
  (*gradient)[0] = ((b[1] - c[1]) * (a[0] * b[1] + b[0] * c[1] + c[0] * a[1] -
                                     a[0] * c[1] - b[0] * a[1] - c[0] * b[1]) +
                    (c[2] - b[2]) * (a[2] * b[0] + b[2] * c[0] + c[2] * a[0] -
                                     a[2] * c[0] - b[2] * a[0] - c[2] * b[0])) /
                   (4 * ar);
  (*gradient)[1] = ((c[0] - b[0]) * (a[0] * b[1] + b[0] * c[1] + c[0] * a[1] -
                                     a[0] * c[1] - b[0] * a[1] - c[0] * b[1]) +
                    (b[2] - c[2]) * (a[1] * b[2] + b[1] * c[2] + c[1] * a[2] -
                                     a[1] * c[2] - b[1] * a[2] - c[1] * b[2])) /
                   (4 * ar);
  (*gradient)[2] = ((b[0] - c[0]) * (a[2] * b[0] + b[2] * c[0] + c[2] * a[0] -
                                     a[2] * c[0] - b[2] * a[0] - c[2] * b[0]) +
                    (c[1] - b[1]) * (a[1] * b[2] + b[1] * c[2] + c[1] * a[2] -
                                     a[1] * c[2] - b[1] * a[2] - c[1] * b[2])) /
                   (4 * ar);
}


// Calculate the gradient (with respect to a) of the area of the triangle
// with points \c p[0], \c p[1] and \c p[2].
template<std::size_t N, typename T>
inline
void
computeGradientOfArea(const std::array<std::array<T, N>, 3>& p,
                      std::array<T, N>* gradient)
{
  return computeGradientOfArea(p[0], p[1], p[2], gradient);
}





//
// Volume.
//

// Return the signed volume of the 3-D tetrahedron with points a, b, c and d.
/*
  Specialization for 3-D.
  The volume is the determinant of
  | 1 a[0] a[1] a[2] |
  | 1 b[0] b[1] b[2] |
  | 1 c[0] c[1] c[2] |
  | 1 d[0] d[1] d[2] |
  (The formula from MathWorld has the wrong sign.)
 */
template<typename T>
inline
T
computeVolume(const std::array<T, 3>& a, const std::array<T, 3>& b,
              const std::array<T, 3>& c, const std::array<T, 3>& d)
{
  return
    (1. / 6) *
    (- a[0] * b[1] * c[2] - a[1] * b[2] * c[0] - a[2] * b[0] * c[1]
     + a[0] * b[2] * c[1] + a[1] * b[0] * c[2] + a[2] * b[1] * c[0]
     + a[0] * b[1] * d[2] + a[1] * b[2] * d[0] + a[2] * b[0] * d[1]
     - a[0] * b[2] * d[1] - a[1] * b[0] * d[2] - a[2] * b[1] * d[0]
     - a[0] * c[1] * d[2] - a[1] * c[2] * d[0] - a[2] * c[0] * d[1]
     + a[0] * c[2] * d[1] + a[1] * c[0] * d[2] + a[2] * c[1] * d[0]
     + b[0] * c[1] * d[2] + b[1] * c[2] * d[0] + b[2] * c[0] * d[1]
     - b[0] * c[2] * d[1] - b[1] * c[0] * d[2] - b[2] * c[1] * d[0]);
}


// Return the signed volume of the tetrahedron with points p[0], p[1], p[2] and p[3].
template<typename T>
inline
T
computeVolume(const std::array<std::array<T, 3>, 4>& p)
{
  return computeVolume(p[0], p[1], p[2], p[3]);
}


// Return the volume of the tetrahedron with points a, b, c and d.
template<std::size_t N, typename T>
inline
T
computeVolume(const std::array<T, N>& a, const std::array<T, N>& b,
              const std::array<T, N>& c, const std::array<T, N>& d)
{
  std::array<std::array<T, N>, 4> p = {{a, b, c, d}};
  return computeVolume(p);
}


// Return the volume of the tetrahedron with points p[0], p[1], p[2] and p[3].
template<std::size_t N, typename T>
inline
T
computeVolume(const std::array<std::array<T, N>, 4>& p)
{
  ads::SquareMatrix<5, T> m;
  m(0, 0) = m(1, 1) = m(2, 2) = m(3, 3) = m(4, 4) = 0;
  m(0, 1) = m(0, 2) = m(0, 3) = m(0, 4) = m(1, 0) = m(2, 0) = m(3, 0) = m(4,
                                          0) = 1;
  for (std::size_t i = 0; i != 3; ++i) {
    for (std::size_t j = i + 1; j != 4; ++j) {
      m(i + 1, j + 1) = m(j + 1, i + 1) = squaredDistance(p[i], p[j]);
    }
  }
  return std::sqrt(ads::computeDeterminant(m) / 288);
}

// Return the volume of the tetrahedron with points \c a, \c b, \c c and \c d.
template<std::size_t N, typename T>
inline
T
computeContent(const std::array<T, N>& a, const std::array<T, N>& b,
               const std::array<T, N>& c, const std::array<T, N>& d)
{
  return computeVolume(a, b, c, d);
}


// Return the volume of the tetrahedron with points \c p[0], \c p[1], \c p[2] and \c p[3].
template<std::size_t N, typename T>
inline
T
computeContent(const std::array<std::array<T, N>, 4>& p)
{
  return computeVolume(p);
}


//
// Gradient of volume.
//


// Calculate the gradient (with respect to a) of the signed volume of the
// 3-D tetrahedron with points a, b, c and d.
template<typename T>
inline
void
computeGradientOfVolume(const std::array<T, 3>& /*a*/,
                        const std::array<T, 3>& b,
                        const std::array<T, 3>& c,
                        const std::array<T, 3>& d,
                        std::array<T, 3>* gradient)
{
  (*gradient)[0] = (b[2] * c[1] - b[1] * c[2] - b[2] * d[1] +
                    c[2] * d[1] + b[1] * d[2] - c[1] * d[2]) / 6;
  (*gradient)[1] = (-b[2] * c[0] + b[0] * c[2] + b[2] * d[0] -
                    c[2] * d[0] - b[0] * d[2] + c[0] * d[2]) / 6;
  (*gradient)[2] = (b[1] * c[0] - b[0] * c[1] - b[1] * d[0] +
                    c[1] * d[0] + b[0] * d[1] - c[0] * d[1]) / 6;
}


// CONTINUE: Implement the gradient of unsigned volume.


} // namespace geom
}
