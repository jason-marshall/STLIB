// -*- C++ -*-

#if !defined(__geom_mesh_simplex_geometry_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom
{

// The dihedral angle between two faces.
template<typename _T>
inline
_T
computeAngle(std::array<std::array<_T, 3>, 3 + 1> const& s,
             std::size_t const a, std::size_t const b) {
  std::array<_T, 3> const an = supportingHyperplane(getFace(s, a)).normal;
  std::array<_T, 3> bn = supportingHyperplane(getFace(s, b)).normal;
  ext::negateElements(&bn);
  return computeAngle(an, bn);
}


// The solid angle at a vertex.
template<typename _T>
inline
_T
computeAngle(const std::array<std::array<_T, 3>, 3 + 1 > & s,
             const std::size_t n) {
#ifdef STLIB_DEBUG
  assert(n < 4);
#endif
  // The simplex dimension plus 1.
  const std::size_t D = 4;
  return computeAngle(s, (n + 1) % D, (n + 2) % D) +
    computeAngle(s, (n + 2) % D, (n + 3) % D) +
    computeAngle(s, (n + 3) % D, (n + 1) % D) - numerical::Constants<_T>::Pi();
}


// The interior angle at a vertex.
template<typename _T>
inline
_T
computeAngle(const std::array<std::array<_T, 2>, 2 + 1 > & s,
             std::size_t n) {
#ifdef STLIB_DEBUG
  assert(n < 3);
#endif
  return computeAngle(s[(n+1)%3] - s[n], s[(n+2)%3] - s[n]);
}

// The interior angle at a vertex is 1.
template<typename _T>
inline
_T
computeAngle(const std::array<std::array<_T, 1>, 1 + 1 > & /*s*/,
             std::size_t /*n*/) {
  return 1;
}

// Project the simplex to a lower dimension.
template<typename _T>
inline
void
projectToLowerDimension(const std::array<std::array<_T, 2>, 1 + 1 > & s,
                        std::array<std::array<_T, 1>, 1 + 1 > * t) {
  // Make the 1-D line segment.
  (*t)[0][0] = 0.0;
  (*t)[1][0] = ext::euclideanDistance(s[0], s[1]);
}

// Project the simplex to a lower dimension.
template<typename _T>
inline
void
projectToLowerDimension(const std::array<std::array<_T, 3>, 1 + 1 > & s,
                        std::array<std::array<_T, 1>, 1 + 1 > * t) {
  // Make the 1-D line segment.
  (*t)[0][0] = 0.0;
  (*t)[1][0] = euclideanDistance(s[0], s[1]);
}


// Project the simplex to a lower dimension.
template<typename _T>
inline
void
projectToLowerDimension(const std::array<std::array<_T, 3>, 2 + 1 > & s,
                        std::array<std::array<_T, 2>, 2 + 1 > * t) {
  typedef std::array<_T, 3> Pt3;

  //
  // Project the triangle in 3-D to a triangle in 2-D.
  //

  // Translate the first vertex to the origin.
  Pt3 v1 = s[1];
  v1 -= s[0];
  Pt3 v2 = s[2];
  v2 -= s[0];

  // Length of 0-1.
  const _T a = ext::magnitude(v1);
  // Length of 0-2.
  const _T b = ext::magnitude(v2);

  // Normalize the vectors that define the edges.
  if (a != 0) {
    v1 /= a;
  }
  if (b != 0) {
    v2 /= b;
  }

  // The cosine of the angle.
  const _T cosTheta = ext::dot(v1, v2);
  // The sine of the angle.
  const _T argument = 1.0 - cosTheta * cosTheta;
  const _T sinTheta = (argument >= 0 ? std::sqrt(argument) : 0);

  // Make the 2-D triangle.
  std::fill((*t)[0].begin(), (*t)[0].end(), 0.0);
  (*t)[1][0] = a;
  (*t)[1][1] = 0.0;
  (*t)[2][0] = b * cosTheta;
  (*t)[2][1] = b * sinTheta;
}

} // namespace geom
} // namespace stlib
