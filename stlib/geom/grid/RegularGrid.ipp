// -*- C++ -*-

/*!
  \file RegularGrid.ipp
  \brief A class for a regular grid.
*/

#if !defined(__geom_RegularGrid_ipp__)
#error This file is an implementation detail of the class RegularGrid.
#endif

namespace stlib
{
namespace geom
{

//
// Constructors
//


template<std::size_t N, typename T>
inline
RegularGrid<N, T>::
RegularGrid(const SizeList& extents, const BBox& domain) :
  _extents(extents),
  _domain(domain),
  _length(domain.upper - domain.lower),
  _delta(),
  _inverseDelta(),
  _indexEpsilon(ext::max(extents)*
                std::numeric_limits<T>::epsilon()),
  _cartesianEpsilon(ext::max(_length) *
                    std::numeric_limits<T>::epsilon())
{
#ifdef STLIB_DEBUG
  for (std::size_t i = 0; i != extents.size(); ++i) {
    assert(extents[i] != 0);
    assert(domain.lower[i] <= domain.upper[i]);
  }
#endif
  Point den;
  for (std::size_t i = 0; i != den.size(); ++i) {
    den[i] = _extents[i] - 1;
  }
  if (ext::product(den) != 0) {
    _delta = _length / den;
  }
  else {
    std::fill(_delta.begin(), _delta.end(), 1);
    _cartesianEpsilon = std::numeric_limits<T>::epsilon();
  }

  for (std::size_t i = 0; i != _inverseDelta.size(); ++i) {
    if (_length[i] != 0) {
      _inverseDelta[i] = den[i] / _length[i];
    }
    else {
      _inverseDelta[i] = std::numeric_limits<T>::max();
    }
  }
}


//
// Equality
//


//! Return true if the true RegularGrid's are equal.
template<std::size_t N, typename T>
inline
bool
operator==(const RegularGrid<N, T>& a, const RegularGrid<N, T>& b)
{
  if (a.getExtents() != b.getExtents()) {
    return false;
  }
  if (a.getDelta() != b.getDelta()) {
    return false;
  }
  if (a.getDomain() != b.getDomain()) {
    return false;
  }
  if (a.getIndexEpsilon() != b.getIndexEpsilon()) {
    return false;
  }
  if (a.getCartesianEpsilon() != b.getCartesianEpsilon()) {
    return false;
  }
  return true;
}


//
// File I/O
//


template<std::size_t N, typename T>
inline
std::ostream&
operator<<(std::ostream& out, const RegularGrid<N, T>& grid)
{
  out << "Dimensions: " << grid.getExtents() << '\n';
  out << "Domain:" << '\n' << grid.getDomain() << '\n';
  return out;
}


//! Read from a file stream.
/*! \relates RegularGrid */
template<std::size_t N, typename T>
inline
std::istream&
operator>>(std::istream& in, RegularGrid<N, T>& grid)
{
  typename RegularGrid<N, T>::SizeList extents;
  typename RegularGrid<N, T>::BBox domain;
  in >> extents >> domain;
  grid = RegularGrid<N, T>(extents, domain);
  return in;
}

} // namespace geom
}
