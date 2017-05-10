// -*- C++ -*-

#if !defined(__geom_simplex_distance_ipp__)
#error This file is an implementation detail of the simplex_distance.
#endif

namespace stlib
{
namespace geom
{

//---------------------------------------------------------------------------
// Inside
//---------------------------------------------------------------------------

template<typename _T>
inline
bool
isIn(const std::array < std::array<_T, 1>, 1 + 1 > & s,
     const std::array<_T, 1>& x) {
  return s[0][0] <= x[0] && x[0] <= s[1][0];
}


template<typename _T>
inline
bool
isIn(const std::array < std::array<_T, 2>, 2 + 1 > & s,
     const std::array<_T, 2>& x) {
  std::array < std::array<_T, 2>, 1 + 1 > face;
  Line_2<_T> line;

  // For each face.
  for (std::size_t n = 0; n != 3; ++n) {
    // Get the vertices of the n_th face.
    getFace(s, n, &face);
    // Make the supporting line of the face.
    line.make(face[0], face[1]);
    if (line.computeSignedDistance(x) > 0) {
      return false;
    }
  }

  // If all of the supporting line distances are non-positive, return true.
  return true;
}


template<typename _T>
inline
bool
isIn(const std::array < std::array<_T, 3>, 3 + 1 > & s,
     const std::array<_T, 3>& x) {
  std::array<std::array<_T, 3>, 2 + 1> face;

  // For each face.
  for (std::size_t n = 0; n != 4; ++n) {
    // Get the vertices of the n_th face.
    getFace(s, n, &face);
    // Make the supporting plane of the face. Then calculate the distance.
    if (signedDistance(supportingHyperplane(face), x) > 0) {
      return false;
    }
  }

  // If all of the supporting plane distances are non-positive, return true.
  return true;
}


//---------------------------------------------------------------------------
// Interior distance.
//---------------------------------------------------------------------------

template<typename _T>
inline
_T
computeDistanceInterior(const std::array < std::array<_T, 1>, 1 + 1 > & s,
                        const std::array<_T, 1>& x) {
#ifdef STLIB_DEBUG
  assert(isIn(s, x));
#endif
  return std::max(s[0][0] - x[0], x[0] - s[1][0]);
}


template<typename _T>
inline
_T
computeDistanceInterior(const std::array < std::array<_T, 2>, 2 + 1 > & s,
                        const std::array<_T, 2>& x) {
  std::array < std::array<_T, 2>, 1 + 1 > face;
  Line_2<_T> line;

  _T d;
  _T distance = - std::numeric_limits<_T>::max();

  // For each face.
  for (std::size_t n = 0; n != 3; ++n) {
    // Get the vertices of the n_th face.
    getFace(s, n, &face);
    // Make the supporting line of the face.
    line.make(face[0], face[1]);
    // Compute the distance to the face.
    d = line.computeSignedDistance(x);
    // Update the distance to the simplex.
    if (d > distance) {
      distance = d;
    }
  }

#ifdef STLIB_DEBUG
  // The distance should be approximately non-positive.
  assert(distance < 10 * std::numeric_limits<_T>::epsilon());
#endif

  // Return the distance to the simplex.
  return distance;
}


template<typename _T>
inline
_T
computeDistanceInterior(const std::array < std::array<_T, 3>, 3 + 1 > & s,
                        const std::array<_T, 3>& x) {
  std::array < std::array<_T, 3>, 2 + 1 > face;

  _T d;
  _T distance = - std::numeric_limits<_T>::max();

  // For each face.
  for (std::size_t n = 0; n != 4; ++n) {
    // Get the vertices of the n_th face.
    getFace(s, n, &face);
    // Make the supporting plane of the face.
    // Compute the distance to the face.
    d = signedDistance(supportingHyperplane(face), x);
    // Update the distance to the simplex.
    if (d > distance) {
      distance = d;
    }
  }

#ifdef STLIB_DEBUG
  // The distance should be approximately non-positive.
  assert(distance < 10 * std::numeric_limits<_T>::epsilon());
#endif

  // Return the distance to the simplex.
  return distance;
}


//---------------------------------------------------------------------------
// Distance.
//---------------------------------------------------------------------------


//! Compute the signed distance from a 1-D point to a line segment.
template<typename _T>
class SimplexDistance<1, 1, _T> :
    public std::unary_function<std::array<_T, 1>, _T> {
  typedef std::unary_function<std::array<_T, 1>, _T> Base;

  std::array<std::array<_T, 1>, 1+1> _s;

public:

  //! Default constructor. Uninitialized data.
  SimplexDistance() {
  }

  //! Construct from the simplex (tetrahedon).
  SimplexDistance(const std::array<std::array<_T, 1>, 1+1>& s) {
    initialize(s);
  }

  //! Initialize with the given simplex.
  void
  initialize(const std::array<std::array<_T, 1>, 1+1>& s) {
    _s = s;
  }

  //! Compute the signed distance from the point to the tetrahedron.
  typename Base::result_type
  operator()(const typename Base::argument_type& x) const {
    if (x[0] < _s[0][0]) {
      return _s[0][0] - x[0];
    }
    if (_s[1][0] < x[0]) {
      return x[0] - _s[1][0];
    }
    return computeDistanceInterior(_s, x);
  }
};

template<typename _T>
inline
_T
computeDistance(const std::array<std::array<_T, 1>, 1+1> & s,
                const std::array<_T, 1>& x) {
  SimplexDistance<1, 1, _T> sd(s);
  return sd(x);
}


//! Compute the unsigned distance from a 2-D point to a line segment.
template<typename _T>
class SimplexDistance<2, 1, _T> :
    public std::unary_function<std::array<_T, 2>, _T> {
  typedef std::unary_function<std::array<_T, 2>, _T> Base;

  std::array<std::array<_T, 2>, 1+1> _s;

  mutable std::array<std::array<_T, 1>, 1+1> _s1;
  mutable std::array<_T, 1> _x1, _y1;

public:

  //! Default constructor. Uninitialized data.
  SimplexDistance() {
  }

  //! Construct from the simplex (line segment).
  SimplexDistance(const std::array<std::array<_T, 2>, 1+1>& s) {
    initialize(s);
  }

  //! Initialize with the given simplex.
  void
  initialize(const std::array<std::array<_T, 2>, 1+1>& s) {
    _s = s;
  }

  //! Compute the unsigned distance from the point to the triangle.
  typename Base::result_type
  operator()(const typename Base::argument_type& x) const {
    project(_s, x, &_s1, &_x1, &_y1);
    _T d = computeUnsignedDistance(_s1, _x1);
    return std::sqrt(d * d + _y1[0] * _y1[0]);
  }
};

template<typename _T>
inline
_T
computeDistance(const std::array<std::array<_T, 2>, 1+1> & s,
                const std::array<_T, 2>& x) {
  SimplexDistance<2, 1, _T> sd(s);
  return sd(x);
}


//! Compute the unsigned distance from a 3-D point to a line segment.
template<typename _T>
class SimplexDistance<3, 1, _T> :
    public std::unary_function<std::array<_T, 3>, _T> {
  typedef std::unary_function<std::array<_T, 3>, _T> Base;

  std::array<std::array<_T, 3>, 1+1> _s;
  std::array<_T, 3> _tangent;

  mutable std::array<_T, 3> _v0;
  mutable std::array<_T, 3> _v1;

public:

  //! Default constructor. Uninitialized data.
  SimplexDistance() {
  }

  //! Construct from the simplex (line segment).
  SimplexDistance(const std::array<std::array<_T, 3>, 1+1>& s) {
    initialize(s);
  }

  //! Initialize with the given simplex.
  void
  initialize(const std::array<std::array<_T, 3>, 1+1>& s) {
    _s = s;
    _tangent = s[1];
    _tangent -= s[0];
    ext::normalize(&_tangent);
  }

  //! Compute the unsigned distance from the point to the triangle.
  typename Base::result_type
  operator()(const typename Base::argument_type& x) const {
    // See if the point is closest to the source end.
    _v0 = x;
    _v0 -= _s[0];
    _T ld0 = ext::dot(_v0, _tangent);
    //std::cerr << "ld0 = " << ld0;
    if (ld0 < 0) {
      return geom::computeDistance(_s[0], x);
    }

    // Next see if the point is closest to the target end.
    _v1 = x;
    _v1 -= _s[1];
    _T ld1 = ext::dot(_v1, _tangent);
    //std::cerr << " ld1 = " << ld1;
    if (ld1 > 0) {
      return geom::computeDistance(_s[1], x);
    }

    // Otherwise, compute the line distance.
    _T d = geom::computeDistance(_s[0], x);
    // With exact arithmetic, the argument of the square root is non-negative,
    // but it may be small and negative due to floating point arithmetic.
    const _T arg = std::max(d * d - ld0 * ld0, _T(0));
    return std::sqrt(arg);
  }
};

template<typename _T>
inline
_T
computeDistance(const std::array<std::array<_T, 3>, 1+1> & s,
                const std::array<_T, 3>& x) {
  SimplexDistance<3, 1, _T> sd(s);
  return sd(x);
}


//! Compute the signed distance from a 2-D point to a triangle.
template<typename _T>
class SimplexDistance<2, 2, _T> :
    public std::unary_function<std::array<_T, 2>, _T> {
  typedef std::unary_function<std::array<_T, 2>, _T> Base;
  typedef std::array<std::array<_T, 2>, 1+1> Face;

  // The faces of the triangle.
  std::array<Face, 3> _faces;
  // The supporting line of a face.
  std::array<Line_2<_T>, 3> _lines;

  // Line distance.
  mutable std::array<_T, 3> _ld;
  // A simplex and point in 1-D.
  mutable std::array < std::array<_T, 1>, 1 + 1 > _s1;
  mutable std::array<_T, 1> _x1;

public:

  //! Default constructor. Uninitialized data.
  SimplexDistance() {
  }

  //! Construct from the simplex (tetrahedon).
  SimplexDistance(const std::array<std::array<_T, 2>, 2+1>& s) {
    initialize(s);
  }

  //! Initialize with the given simplex.
  void
  initialize(const std::array<std::array<_T, 2>, 2+1>& s) {
    // For each vertex/face.
    for (std::size_t n = 0; n != 3; ++n) {
      // Get the vertices of the n_th face.
      getFace(s, n, &_faces[n]);
      // Make the supporting line of the face.
      _lines[n].make(_faces[n][0], _faces[n][1]);
    }
  }

  //! Compute the signed distance from the point to the tetrahedron.
  typename Base::result_type
  operator()(const typename Base::argument_type& x) const {
    // Compute the distance to each supporting line.
    for (std::size_t n = 0; n != 3; ++n) {
      _ld[n] = _lines[n].computeSignedDistance(x);
    }
    // First check the case that the point is inside the triangle.
    _T d = ext::max(_ld);
    if (d <= 0) {
      return d;
    }
    // Next compute the distance to each face for which the point is above
    // the supporting line.
    for (std::size_t n = 0; n != 3; ++n) {
      // If above the n_th face.
      if (_ld[n] > 0) {
        // Return the distance to the line segment.
        project(_faces[n], x, &_s1, &_x1);
        d = computeUnsignedDistance(_s1, _x1);
        _ld[n] = std::sqrt(d * d + _ld[n] * _ld[n]);
      }
      else {
        _ld[n] = std::numeric_limits<_T>::max();
      }
    }
    // Return the distance to the closest line segment.
    return ext::min(_ld);
  }
};

template<typename _T>
inline
_T
computeDistance(const std::array<std::array<_T, 2>, 2+1> & s,
                const std::array<_T, 2>& x) {
  SimplexDistance<2, 2, _T> sd(s);
  return sd(x);
}

//! Compute the unsigned distance from a 3-D point to a triangle.
template<typename _T>
class SimplexDistance<3, 2, _T> :
    public std::unary_function<std::array<_T, 3>, _T> {
  typedef std::unary_function<std::array<_T, 3>, _T> Base;

  mutable std::array<std::array<_T, 3>, 2+1> _s;
  mutable std::array<std::array<_T, 2>, 2+1> _s2;
  mutable std::array<_T, 2> _x2;
  mutable std::array<_T, 1> _z1;

public:

  //! Default constructor. Uninitialized data.
  SimplexDistance() {
  }

  //! Construct from the simplex (triangle).
  SimplexDistance(const std::array<std::array<_T, 3>, 2+1>& s) {
    initialize(s);
  }

  //! Initialize with the given simplex.
  void
  initialize(const std::array<std::array<_T, 3>, 2+1>& s) {
    _s = s;
  }

  //! Compute the unsigned distance from the point to the triangle.
  typename Base::result_type
  operator()(const typename Base::argument_type& x) const {
    project(_s, x, &_s2, &_x2, &_z1);
    _T d = computeUnsignedDistance(_s2, _x2);
    return std::sqrt(d * d + _z1[0] * _z1[0]);
  }
};

template<typename _T>
inline
_T
computeDistance(const std::array<std::array<_T, 3>, 2+1> & s,
                const std::array<_T, 3>& x) {
  SimplexDistance<3, 2, _T> sd(s);
  return sd(x);
}


//! Compute the signed distance from a 3-D point to a tetrahedron.
template<typename _T>
class SimplexDistance<3, 3, _T> :
    public std::unary_function<std::array<_T, 3>, _T> {
  typedef std::unary_function<std::array<_T, 3>, _T> Base;
  typedef std::array<std::array<_T, 3>, 2 + 1> Face;

  // The faces of the tetrahedron.
  std::array<Face, 4> _faces;
  // The supporting planes of the faces.
  std::array<Hyperplane<_T, 3>, 4> _planes;

  // Plane distance.
  mutable std::array<_T, 4> _pd;
  // A simplex and point in 2-D.
  mutable std::array<std::array<_T, 2>, 2+1> _s2;
  mutable std::array<_T, 2> _x2;

public:

  //! Default constructor. Uninitialized data.
  SimplexDistance() {
  }

  //! Construct from the simplex (tetrahedon).
  SimplexDistance(const std::array<std::array<_T, 3>, 3+1>& s) {
    initialize(s);
  }

  //! Initialize with the given simplex.
  void
  initialize(const std::array<std::array<_T, 3>, 3+1>& s) {
    // For each vertex/face.
    for (std::size_t n = 0; n != 4; ++n) {
      // Get the vertices of the n_th face.
      getFace(s, n, &_faces[n]);
      // Make the supporting plane of the face.
      _planes[n] = supportingHyperplane(_faces[n]);
    }
  }

  //! Compute the signed distance from the point to the tetrahedron.
  typename Base::result_type
  operator()(const typename Base::argument_type& x) const {
    // Compute the distance to each plane.
    for (std::size_t n = 0; n != 4; ++n) {
      _pd[n] = signedDistance(_planes[n], x);
    }
    // First check the case that the point is inside the tet.
    _T d = ext::max(_pd);
    if (d <= 0) {
      return d;
    }
    // Next compute the distance to each face for which the point is above
    // the supporting plane.
    for (std::size_t n = 0; n != 4; ++n) {
      // If above the n_th face.
      if (_pd[n] > 0) {
        // Return the distance to the triangle.
        project(_faces[n], x, &_s2, &_x2);
        d = computeUnsignedDistance(_s2, _x2);
        _pd[n] = std::sqrt(d * d + _pd[n] * _pd[n]);
      }
      else {
        _pd[n] = std::numeric_limits<_T>::max();
      }
    }
    // Return the distance to the closest triangle face.
    return ext::min(_pd);
  }
};

template<typename _T>
inline
_T
computeDistance(const std::array<std::array<_T, 3>, 3+1> & s,
                const std::array<_T, 3>& x) {
  SimplexDistance<3, 3, _T> sd(s);
  return sd(x);
}


//---------------------------------------------------------------------------
// Signed distance.
//---------------------------------------------------------------------------


template <std::size_t N, typename _T>
inline
_T
computeSignedDistance(const std::array<_T, N>& p,
                      const std::array<_T, N>& n,
                      const std::array<_T, N>& x) {
  std::array<_T, N> v;
  v = x;
  v -= p;

  _T d = geom::computeDistance(p, x);

  if (ext::dot(v, n) > 0) {
    return d;
  }
  return -d;
}


template<typename _T>
inline
_T
computeSignedDistance(const std::array < std::array<_T, 2>, 1 + 1 > & s,
                      const std::array<_T, 2>& x) {
  std::array < std::array<_T, 1>, 1 + 1 > s1;
  std::array<_T, 1> x1, y1;

  project(s, x, &s1, &x1, &y1);
  _T d = computeUnsignedDistance(s1, x1);
  if (d <= 0) {
    // Return the signed distance.
    return -y1[0];
  }
  return std::numeric_limits<_T>::max();
}


template<typename _T>
inline
_T
computeSignedDistance(const std::array < std::array<_T, 2>, 1 + 1 > & s,
                      const std::array<_T, 2>& x,
                      std::array<_T, 2>* closestPoint) {
  std::array < std::array<_T, 1>, 1 + 1 > s1;
  std::array<_T, 1> x1, y1;

  project(s, x, &s1, &x1, &y1);
  _T d = computeUnsignedDistance(s1, x1);
  if (d <= 0) {
    // First compute the tangent.
    *closestPoint = s[1];
    *closestPoint -= s[0];
    ext::normalize(closestPoint);
    // Then the closest point.
    *closestPoint *= x1[0];
    *closestPoint += s[0];
    // Return the signed distance.
    return -y1[0];
  }
  return std::numeric_limits<_T>::max();
}



template<typename _T>
inline
_T
computeSignedDistance(const std::array < std::array<_T, 3>, 2 + 1 > & s,
                      const std::array<_T, 3>& x) {
  std::array < std::array<_T, 2>, 2 + 1 > s2;
  std::array<_T, 2> x2;
  std::array<_T, 1> z1;

  project(s, x, &s2, &x2, &z1);
  _T d = computeUnsignedDistance(s2, x2);
  if (d <= 0) {
    return z1[0];
  }
  return std::numeric_limits<_T>::max();
}


template<typename _T>
inline
_T
computeSignedDistance(const std::array < std::array<_T, 3>, 2 + 1 > & s,
                      const std::array<_T, 3>& n,
                      const std::array<_T, 3>& x,
                      std::array<_T, 3>* closestPoint) {
  std::array < std::array<_T, 2>, 2 + 1 > s2;
  std::array<_T, 3> offset;
  std::array<_T, 2> x2;
  std::array<_T, 1> z1;

  project(s, x, &s2, &x2, &z1);
  const _T d = computeUnsignedDistance(s2, x2);
  // If the point is directly above of below the triangle face.
  if (d <= 0) {
    // For the closest point on the triangle face, start at the point x.
    *closestPoint = x;
    // Then subtract the normal to the face times the distance from the face.
    offset = n;
    offset *= z1[0];
    *closestPoint -= offset;
    // Return the signed distance.
    return z1[0];
  }
  return std::numeric_limits<_T>::max();
}


template<typename _T>
inline
_T
computeSignedDistance(const std::array < std::array<_T, 3>, 1 + 1 > & s,
                      const std::array<_T, 3>& n,
                      const std::array<_T, 3>& x) {
  std::array<_T, 3> tangent, v0, v1;
  tangent = s[1];
  tangent -= s[0];
  ext::normalize(&tangent);

  // See if the point is closest to the source end.
  v0 = x;
  v0 -= s[0];
  _T ld0 = ext::dot(v0, tangent);
  if (ld0 < 0) {
    return std::numeric_limits<_T>::max();
  }

  // Next see if the point is closest to the target end.
  v1 = x;
  v1 -= s[1];
  _T ld1 = ext::dot(v1, tangent);
  if (ld1 > 0) {
    return std::numeric_limits<_T>::max();
  }

  // Otherwise, compute the distance from the supporting line.
  _T d = geom::computeDistance(s[0], x);
  // With exact arithmetic, the argument of the square root is non-negative,
  // but it may be small and negative due to floating point arithmetic.
  const _T arg = std::max(d * d - ld0 * ld0, 0.);
  d = std::sqrt(arg);

  // Return the signed distance.
  if (ext::dot(v0, n) > 0) {
    return d;
  }
  return -d;
}


template<typename _T>
inline
_T
computeSignedDistance(const std::array < std::array<_T, 3>, 1 + 1 > & s,
                      const std::array<_T, 3>& n,
                      const std::array<_T, 3>& x,
                      std::array<_T, 3>* closestPoint) {
  std::array<_T, 3> tangent, v0, v1;
  tangent = s[1];
  tangent -= s[0];
  ext::normalize(&tangent);

  // See if the point is closest to the source end.
  v0 = x;
  v0 -= s[0];
  _T ld0 = ext::dot(v0, tangent);
  if (ld0 < 0) {
    return std::numeric_limits<_T>::max();
  }

  // Next see if the point is closest to the target end.
  v1 = x;
  v1 -= s[1];
  _T ld1 = ext::dot(v1, tangent);
  if (ld1 > 0) {
    return std::numeric_limits<_T>::max();
  }

  // Otherwise, compute the distance from the supporting line.
  _T d = geom::computeDistance(s[0], x);
  // With exact arithmetic, the argument of the square root is non-negative,
  // but it may be small and negative due to floating point arithmetic.
  const _T arg = std::max(d * d - ld0 * ld0, 0.);
  d = std::sqrt(arg);

  // To compute the closest point, we start at the source vertex and add
  // the line distance times the tangent.
  *closestPoint = s[0];
  tangent *= ld0;
  *closestPoint += tangent;

  // Return the signed distance.
  if (ext::dot(v0, n) > 0) {
    return d;
  }
  return -d;
}


//---------------------------------------------------------------------------
// Project to a lower dimension.
//---------------------------------------------------------------------------


// Project the simplex and the point in 2-D to 1-D.
template<typename _T>
inline
void
project(const std::array < std::array<_T, 2>, 1 + 1 > & s2,
        const std::array<_T, 2>& x2,
        std::array < std::array<_T, 1>, 1 + 1 > * s1,
        std::array<_T, 1>* x1) {
  // We don't use the y offset.
  std::array<_T, 1> y1;
  project(s2, x2, s1, x1, &y1);
}



// Project the simplex and the point in 2-D to 1-D.
template<typename _T>
inline
void
project(const std::array < std::array<_T, 2>, 1 + 1 > & s2,
        const std::array<_T, 2>& x2,
        std::array < std::array<_T, 1>, 1 + 1 > * s1,
        std::array<_T, 1>* x1,
        std::array<_T, 1>* y1) {
  // A 2-D Cartesian point.
  typedef std::array<_T, 2> P2;

  P2 tangent, x;

  //
  // We use the line segment tangent to determine the mapping.
  //

  // Convert the line segment to a vector.
  tangent = s2[1];
  tangent -= s2[0];
  // The length of the vector.
  const _T mag = ext::magnitude(tangent);
  // The unit tangent to the line segment.
  if (mag != 0) {
    tangent /= mag;
  }
  else {
    // Degenerate case: zero length simplex.
    tangent[0] = 1;
    tangent[1] = 0;
  }

  //
  // Map the simplex in 2-D to a simplex in 1-D
  //

  // The first vertex of the mapped simplex is the origin.
  (*s1)[0][0] = 0;
  // The second vertex of the mapped simplex lies on the positive x axis.
  (*s1)[1][0] = mag;

  // Copy the 2-D point.
  x = x2;
  // Translate the point by the same amount the simplex was translated.
  x -= s2[0];
  // Rotate the point.
  (*x1)[0] = x[0] * tangent[0] + x[1] * tangent[1];
  (*y1)[0] = - x[0] * tangent[1] + x[1] * tangent[0];
  // (c  s) (x[0])
  // (-s c) (x[1])
  // c = tangent[0]
  // s = tangent[1]
}



// Project the simplex and the point in 3-D to 2-D.
template<typename _T>
inline
void
project(const std::array < std::array<_T, 3>, 2 + 1 > & s3,
        const std::array<_T, 3>& x3,
        std::array < std::array<_T, 2>, 2 + 1 > * s2,
        std::array<_T, 2>* x2) {
  // We don't use the z offset.
  std::array<_T, 1> z1;
  project(s3, x3, s2, x2, &z1);
}



// Project the simplex and the point in 3-D to 2-D.
template<typename _T>
inline
void
project(const std::array < std::array<_T, 3>, 2 + 1 > & s3,
        const std::array<_T, 3>& x3,
        std::array < std::array<_T, 2>, 2 + 1 > * s2,
        std::array<_T, 2>* x2,
        std::array<_T, 1>* z1) {
  typedef std::array<_T, 3> P3;
  typedef ads::SquareMatrix<3, _T> Matrix;

  Matrix inverseMapping, mapping;
  P3 xi, psi, zeta, a, b;

  //
  // First determine the inverse mapping, (x,y,z)->(xi,psi,zeta).
  //

  // xi is the tangent to the first edge of the triangle.
  xi = s3[1];
  xi -= s3[0];
  ext::normalize(&xi);

  // zeta is normal to the triangle.
  a = s3[2];
  a -= s3[0];
  ext::cross(xi, a, &zeta);
  ext::normalize(&zeta);

  ext::cross(zeta, xi, &psi);

  inverseMapping.set(xi[0], psi[0], zeta[0],
                     xi[1], psi[1], zeta[1],
                     xi[2], psi[2], zeta[2]);

  // Take the inverse to get the mapping.  Since this is a rotation,
  // the determinant of the matrix is 1.
  ads::computeInverse(inverseMapping, _T(1), &mapping);

  // The first point is mapped to the 2-D origin.
  (*s2)[0].fill(0);
  // The second point is mapped to the x axis.
  a = s3[1];
  a -= s3[0];
  ads::computeProduct(mapping, a, &b);
  (*s2)[1][0] = b[0];
  (*s2)[1][1] = b[1];
  // The third point is mapped to the xy plane.
  a = s3[2];
  a -= s3[0];
  ads::computeProduct(mapping, a, &b);
  (*s2)[2][0] = b[0];
  (*s2)[2][1] = b[1];
  // Finally, map the free point.
  a = x3;
  a -= s3[0];
  ads::computeProduct(mapping, a, &b);
  (*x2)[0] = b[0];
  (*x2)[1] = b[1];
  (*z1)[0] = b[2];
}





// One can map a 2-simplex in 3-D to a 2-simplex in 2-D with a translation
// and a rotation.
// Compute the rotation (and its inverse) for this mapping from the
// 2-simplex in 3-D to a 2-simplex in 2-D.
template<typename _T>
inline
void
computeRotation(const std::array < std::array<_T, 3>, 2 + 1 > & simplex,
                ads::SquareMatrix<3, _T>* rotation,
                ads::SquareMatrix<3, _T>* inverseRotation) {
  typedef std::array<_T, 3> Point;

  Point xi, psi, zeta;

  //
  // First determine the inverse mapping, (x,y,z)->(xi,psi,zeta).
  //

  // xi is the tangent to the first edge of the triangle.
  xi = simplex[1];
  xi -= simplex[0];
  ext::normalize(&xi);

  // zeta is normal to the triangle.
  psi = simplex[2]; // Use psi as a temporary variable.
  psi -= simplex[0];
  ext::cross(xi, psi, &zeta);
  ext::normalize(&zeta);

  // psi is orthonormal to zeta and xi.
  ext::cross(zeta, xi, &psi);

  inverseRotation->set(xi[0], psi[0], zeta[0],
                       xi[1], psi[1], zeta[1],
                       xi[2], psi[2], zeta[2]);

  // Take the inverse to get the mapping. Since this is a rotation,
  // the determinant of the matrix is 1.
  ads::computeInverse(*inverseRotation, _T(1), rotation);
}



template<typename _T>
inline
void
mapSimplex(const std::array < std::array<_T, 3>, 2 + 1 > & s3,
           const ads::SquareMatrix<3, _T>& rotation,
           std::array < std::array<_T, 2>, 2 + 1 > * s2) {
  typedef std::array<_T, 3> P3;

  P3 a, b;

  // The first point is mapped to the 2-D origin.
  (*s2)[0].fill(0);
  // The second point is mapped to the x axis.
  a = s3[1];
  a -= s3[0];
  ads::computeProduct(rotation, a, &b);
  (*s2)[1][0] = b[0];
  (*s2)[1][1] = b[1];
  // The third point is mapped to the xy plane.
  a = s3[2];
  a -= s3[0];
  ads::computeProduct(rotation, a, &b);
  (*s2)[2][0] = b[0];
  (*s2)[2][1] = b[1];
}



template<typename _T>
inline
void
mapPointDown(const std::array<_T, 3>& x3,
             const std::array < std::array<_T, 3>, 2 + 1 > & s3,
             const ads::SquareMatrix<3, _T>& rotation,
             std::array<_T, 2>* x2) {
  typedef std::array<_T, 3> P3;

  P3 a, b;

  // Translate.
  a = x3;
  a -= s3[0];
  // Rotate.
  ads::computeProduct(rotation, a, &b);
  (*x2)[0] = b[0];
  (*x2)[1] = b[1];
  //(*z1)[0] = b[2];
}



template<typename _T>
inline
void
mapPointUp(const std::array<_T, 2>& x2,
           const std::array < std::array<_T, 3>, 2 + 1 > & s3,
           const ads::SquareMatrix<3, _T>& inverseRotation,
           std::array<_T, 3>* x3) {
  typedef std::array<_T, 3> P3;

  // Rotate.
  P3 a;
  a[0] = x2[0];
  a[1] = x2[1];
  a[2] = 0;
  ads::computeProduct(inverseRotation, a, x3);
  // Translate.
  *x3 += s3[0];
}



//---------------------------------------------------------------------------
// Closest Point.
//---------------------------------------------------------------------------

// Return the unsigned distance from the 2-D point to the 1-simplex
// and compute the closest point.
template<typename _T>
inline
_T
computeClosestPoint(const std::array < std::array<_T, 2>, 1 + 1 > & simplex,
                    const std::array<_T, 2>& point,
                    std::array<_T, 2>* closestPoint) {
  geom::SegmentMath<2, _T> segment(simplex[0], simplex[1]);
  return computeDistanceAndClosestPoint(segment, point, closestPoint);
}



// Return the unsigned distance from the 3-D point to the 1-simplex
// and compute the closest point.
template<typename _T>
inline
_T
computeClosestPoint(const std::array < std::array<_T, 3>, 1 + 1 > & simplex,
                    const std::array<_T, 3>& point,
                    std::array<_T, 3>* closestPoint) {
  geom::SegmentMath<3, _T> segment(simplex[0], simplex[1]);
  return computeDistanceAndClosestPoint(segment, point, closestPoint);
}



// Return the unsigned distance from the 2-D point to the 2-simplex
// and compute the closest point.
template<typename _T>
inline
_T
computeClosestPoint(const std::array < std::array<_T, 2>, 2 + 1 > & simplex,
                    const std::array<_T, 2>& point,
                    std::array<_T, 2>* closestPoint) {
  typedef std::array<_T, 2> Point2;
  typedef std::array < Point2, 1 + 1 > Face;

  // The simplex dimension.
  const std::size_t M = 2;

  // A face of the triangle.
  Face face;
  // The supporting line of a face.
  Line_2<_T> line;
  // Line distance.
  std::array < _T, M + 1 > lineDistance;

  // For each vertex/face.
  for (int m = 0; m != M + 1; ++m) {
    // Get the vertices of the m_th face.
    getFace(simplex, m, &face);
    // Make the supporting line of the face.
    line.make(face[0], face[1]);
    // Compute the signed distance to the line.
    lineDistance[m] = line.computeSignedDistance(point);
    // If above the m_th face.  (Outside the triangle.)
    if (lineDistance[m] > 0) {
      // Compute the closest point on the line segment.
      // Return the positive distance to the line segment.
      return computeClosestPoint(face, point, closestPoint);
    }
  }

  // If the point is below all faces, the distance is non-positive.
  *closestPoint = point;
  return ext::max(lineDistance);
}



// Return the unsigned distance from the 3-D point to the 2-simplex
// and compute the closest point.
template<typename _T>
inline
_T
computeClosestPoint(const std::array < std::array<_T, 3>, 2 + 1 > & simplex,
                    const std::array<_T, 3>& point,
                    std::array<_T, 3>* closestPoint) {
  typedef std::array<_T, 2> Point2;
  typedef std::array < Point2, 2 + 1 > Simplex2;

  // Determine the rotations in the mapping between 3-D and 2-D.
  ads::SquareMatrix<3, _T> rotation, inverseRotation;
  computeRotation(simplex, &rotation, &inverseRotation);

  // Map the simplex.
  Simplex2 simplex2;
  mapSimplex(simplex, rotation, &simplex2);

  // Map the point down to 2-D.
  Point2 point2;
  mapPointDown(point, simplex, rotation, &point2);

  // Compute the closest point in 2-D.
  Point2 closestPoint2;
  computeClosestPoint(simplex2, point2, &closestPoint2);

  // Map the point up to 3-D.
  mapPointUp(closestPoint2, simplex, inverseRotation, closestPoint);

  // Return the unsigned distance.
  return geom::computeDistance(point, *closestPoint);
}

} // namespace geom
} // namespace stlib
