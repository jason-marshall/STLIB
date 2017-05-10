// -*- C++ -*-

#if !defined(__geom_Hyperplane_ipp__)
#error This file is an implementation detail of the class Hyperplane.
#endif

namespace stlib
{
namespace geom
{

//
// Mathematical member functions
//

template<typename _T, std::size_t _D>
inline
_T
signedDistance(Hyperplane<_T, _D> const& hyperplane,
               typename Hyperplane<_T, _D>::Point const& p,
               typename Hyperplane<_T, _D>::Point* closestPoint)
{
#ifdef STLIB_DEBUG
  assert(isValid(hyperplane));
#endif
  _T const dist = signedDistance(hyperplane, p);
  //*closestPoint = p - dist * hyperplane.normal;
  for (std::size_t i = 0; i != p.size(); ++i) {
    (*closestPoint)[i] = p[i] - dist * hyperplane.normal[i];
  }
  return dist;
}


//
// File IO
//


template<typename _T, std::size_t _D>
inline
std::istream&
operator>>(std::istream& in, Hyperplane<_T, _D>& x)
{
  in >> x.point >> x.normal;
#ifdef STLIB_DEBUG
  assert(isValid(x));
#endif
  return in;
}


template<typename _T, std::size_t _D>
inline
std::ostream&
operator<<(std::ostream& out, Hyperplane<_T, _D> const& x)
{
  return out << x.point << ' ' << x.normal;
}


template<typename _T>
inline
Hyperplane<_T, 1>
supportingHyperplane(std::array<std::array<_T, 1>, 2> const& simplex,
                     std::size_t const n)
{
#ifdef STLIB_DEBUG
  assert(n < 2);
#endif
  if (simplex[0] == simplex[1]) {
    throw std::runtime_error
      ("In stlib::geom::supportingHyperplane(): The two vertices of the "
       "simplex are equal. Cannot define the supporting hyperplane.");
  }
   
  std::size_t const a = (n + 1) % 2;
  std::array<_T, 1> normal = simplex[a] - simplex[n];
  ext::normalize(&normal);
  return Hyperplane<_T, 1>{simplex[a], normal};
}


template<typename _T>
inline
Hyperplane<_T, 2>
supportingHyperplane(std::array<std::array<_T, 2>, 2> const& face)
{
  if (face[0] == face[1]) {
    throw std::runtime_error
      ("In stlib::geom::supportingHyperplane(): The two vertices of the "
       "simplex face are equal. Cannot define the supporting hyperplane.");
  }
  std::array<_T, 2> normal = face[1] - face[0];
  rotateMinusPiOver2(&normal);
  ext::normalize(&normal);
  return Hyperplane<_T, 2>{face[0], normal};
}


template<typename _T>
inline
Hyperplane<_T, 2>
supportingHyperplane(std::array<std::array<_T, 2>, 3> const& simplex,
                     std::size_t const n)
{
#ifdef STLIB_DEBUG
  assert(n < 3);
#endif
  // This formula defines the correctly oriented face.
  std::size_t const a = (n + 1) % 3;
  std::size_t const b = (n + 2) % 3;
  return supportingHyperplane(std::array<std::array<_T, 2>, 2>
                              {{simplex[a], simplex[b]}});
}


template<typename _T>
inline
Hyperplane<_T, 3>
supportingHyperplane(std::array<std::array<_T, 3>, 3> const& face)
{
  std::array<_T, 3> normal = ext::cross(face[1] - face[0], face[2] - face[0]);
  _T const m2 = ext::squaredMagnitude(normal);
  if (m2 == 0) {
    throw std::runtime_error
      ("In stlib::geom::supportingHyperplane(): The two vectors in the plane "
       "are not linearly independent. Cannot calculate the normal.");
  }
  normal /= std::sqrt(m2);
  return Hyperplane<_T, 3>{face[0], normal};
}


template<typename _T>
inline
Hyperplane<_T, 3>
supportingHyperplane(std::array<std::array<_T, 3>, 4> const& simplex,
                     std::size_t const n)
{
  return supportingHyperplane(getFace(simplex, n));
}


} // namespace geom
} // namespace stlib
