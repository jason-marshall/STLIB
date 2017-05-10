// -*- C++ -*-

/*!
  \file SimpleRegularGrid.ipp
  \brief A class for a regular grid.
*/

#if !defined(__geom_SimpleRegularGrid_ipp__)
#error This file is an implementation detail of the class SimpleRegularGrid.
#endif

namespace stlib
{
namespace geom
{

//
// Constructors
//


template<typename _T, std::size_t _Dimension>
inline
SimpleRegularGrid<_T, _Dimension>::
SimpleRegularGrid(const IndexList& extents, const BBox& domain) :
  _extents(extents),
  _lower(domain.lower),
  _delta(),
  _inverseDelta()
{
  // Lengths of the sides of the box.
  const Point lengths = domain.upper - domain.lower;

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
    _delta = lengths / den;
  }
  else {
    std::fill(_delta.begin(), _delta.end(), Number(1));
  }

  for (std::size_t i = 0; i != _inverseDelta.size(); ++i) {
    if (lengths[i] != 0) {
      _inverseDelta[i] = den[i] / lengths[i];
    }
    else {
      _inverseDelta[i] = 1;
    }
  }
}


//
// Equality
//


//! Return true if the true SimpleRegularGrid's are equal.
template<typename _T, std::size_t _Dimension>
inline
bool
operator==(const SimpleRegularGrid<_T, _Dimension>& a,
           const SimpleRegularGrid<_T, _Dimension>& b)
{
  if (a.getExtents() != b.getExtents()) {
    return false;
  }
  if (a.getLower() != b.getLower()) {
    return false;
  }
  if (a.getDelta() != b.getDelta()) {
    return false;
  }
  return true;
}


//
// File I/O
//


template<typename _T, std::size_t _Dimension>
inline
std::ostream&
operator<<(std::ostream& out, const SimpleRegularGrid<_T, _Dimension>& x)
{
  return out << x.getExtents() << '\n'
         << x.getLower() << '\n'
         << x.getDelta() << '\n';
}


template<typename _T, std::size_t _Dimension>
inline
std::istream&
operator>>(std::istream& in, SimpleRegularGrid<_T, _Dimension>& x)
{
  in >> x._extents >> x._lower >> x._delta;
  for (std::size_t i = 0; i != _Dimension; ++i) {
    x._inverseDelta[i] = 1. / x._delta[i];
  }
  return in;
}

} // namespace geom
}
