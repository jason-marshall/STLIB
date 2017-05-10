// -*- C++ -*-

#if !defined(__geom_spatialIndexing_SpecialEuclideanCodeUniformGrid_ipp__)
#error This file is an implementation detail of the class SpecialEuclideanCodeUniformGrid.
#endif

namespace stlib
{
namespace geom
{


template<typename _Key>
inline
typename SpecialEuclideanCodeUniformGrid<3, _Key>::Key
SpecialEuclideanCodeUniformGrid<3, _Key>::
encode(const Quaternion& q, const Point& translation) const
{
#ifdef STLIB_DEBUG
  // Check that q is a unit quaternion.
  assert(numerical::areEqual(norm(q), 1., 10.));
#endif
  // Rotate the x and y axes as quaternions, and record them as vectors.
  Quaternion xq(0, 1, 0, 0);
  xq = q * xq * conj(q);
  const Point x = {{
      xq.R_component_2(), xq.R_component_3(),
      xq.R_component_4()
    }
  };
  Quaternion yq(0, 0, 1, 0);
  yq = q * yq * conj(q);
  const Point y = {{
      yq.R_component_2(), yq.R_component_3(),
      yq.R_component_4()
    }
  };
  // Use the rotated axes to encode the transformation.
  return encode(x, y, translation);
}


template<typename _Key>
inline
void
SpecialEuclideanCodeUniformGrid<3, _Key>::
decode(Key key, Quaternion* q, Point* translation) const
{
  // Extract the rotated x and y axes and the translation from the key.
  std::array<Point, 3> ax;
  decode(key, &ax[0], &ax[1], translation);
  // Now we need to convert the rotation, represented with the two
  // rotated axes, into a quaternion representation. To do this we first
  // construct the rotation matrix, and then convert that to a quaternion.
  // The rotation matrix has the transformed axes as columns.
  ext::cross(ax[0], ax[1], &ax[2]);
  // r(i, j) == ax[j][i]
  const double trace = ax[0][0] + ax[1][1] + ax[2][2];
  if (trace > 0) {
    const double w = 0.5 * std::sqrt(trace + 1);
    const double fac = 1. / (4 * w);
    *q = Quaternion(w, (ax[1][2] - ax[2][1]) * fac,
                    (ax[0][2] - ax[2][0]) * fac,
                    (ax[0][1] - ax[1][0]) * fac);
  }
  else {
    // Pick the largest component of the axis of rotation.
    if (ax[0][0] > ax[1][1] && ax[0][0] > ax[2][2]) {
      const double x = 0.5 * std::sqrt(ax[0][0] - ax[1][1] - ax[2][2] + 1);
      const double fac = 1. / (4 * x);
      *q = Quaternion((ax[2][1] - ax[1][2]) * fac,
                      x,
                      (ax[1][0] + ax[0][1]) * fac,
                      (ax[2][0] + ax[0][2]) * fac);
    }
    else if (ax[1][1] > ax[0][0] && ax[1][1] > ax[2][2]) {
      const double y = 0.5 * std::sqrt(ax[1][1] - ax[0][0] - ax[2][2] + 1);
      const double fac = 1. / (4 * y);
      *q = Quaternion((ax[0][2] - ax[2][0]) * fac,
                      (ax[1][0] + ax[0][1]) * fac,
                      y,
                      (ax[2][1] + ax[1][2]) * fac);
    }
    else {
      const double z = 0.5 * std::sqrt(ax[2][2] - ax[0][0] - ax[1][1] + 1);
      const double fac = 1. / (4 * z);
      *q = Quaternion((ax[1][0] - ax[0][1]) * fac,
                      (ax[2][0] + ax[0][2]) * fac,
                      (ax[2][1] + ax[1][2]) * fac,
                      z);
    }
  }
}


template<typename _Key>
inline
SpecialEuclideanCodeUniformGrid<3, _Key>::
SpecialEuclideanCodeUniformGrid(const BBox& domain, const double spacing,
                                const std::size_t bitsPerRotationCoordinate) :
  _lower(),
  _spacing(spacing),
  _bitsPerTranslation(),
  _bitsPerRotation(bitsPerRotationCoordinate)
{
  const double Eps = std::numeric_limits<double>::epsilon();
  // Determine the number of bits to use for each of the translation
  // coordinates.
  for (std::size_t d = 0; d != D; ++d) {
    // The number of cells needed to span the domain in this coordinate.
    std::size_t n = std::size_t((domain.upper[d] - domain.lower[d]) *
                                (1 + Eps) / spacing) + 1;
    // Calculate the number of required bits for this number of cells.
    n -= 1;
    std::size_t e = 0;
    while (n) {
      n >>= 1;
      ++e;
    }
    _bitsPerTranslation[d] = e;
  }
  assert(2 * D * _bitsPerRotation + ext::sum(_bitsPerTranslation) <=
         std::size_t(std::numeric_limits<_Key>::digits));
  // From the number of bits, determine the cell extents.
  std::array<std::size_t, D> extents;
  for (std::size_t i = 0; i != D; ++i) {
    extents[i] = 1 << _bitsPerTranslation[i];
  }
  // Determine the lower bound.
  _lower = 0.5 * (domain.lower + domain.upper) -
    (0.5 * spacing) * ext::convert_array<double>(extents);
}

template<typename _Key>
inline
typename SpecialEuclideanCodeUniformGrid<3, _Key>::Key
SpecialEuclideanCodeUniformGrid<3, _Key>::
encode(const Point& x, const Point& y, const Point& translation) const
{
#ifdef STLIB_DEBUG
  // x and y must be normal.
  assert(numerical::areEqual(ext::squaredMagnitude(x), 1., 10.));
  assert(numerical::areEqual(ext::squaredMagnitude(y), 1., 10.));
#endif
  Key key = 0;
  //
  // First pack the rotation.
  //
  const std::size_t extents = 1 << _bitsPerRotation;
  const double rotationSpacing = 2. / (extents - 1);
  const double inverseRotationSpacing = 1. / rotationSpacing;
  // The rotated x axis.
  for (std::size_t i = 0; i != D; ++i) {
    const std::size_t n = std::size_t((x[i] + 1 + 0.5 * rotationSpacing) *
                                      inverseRotationSpacing);
    key <<= _bitsPerRotation;
    key |= n;
  }
  // The rotated y axis.
  for (std::size_t i = 0; i != D; ++i) {
    const std::size_t n = std::size_t((y[i] + 1 + 0.5 * rotationSpacing) *
                                      inverseRotationSpacing);
    key <<= _bitsPerRotation;
    key |= n;
  }

  //
  // Then pack the translation.
  //
  const double inverseSpacing = 1. / _spacing;
  for (std::size_t i = 0; i != D; ++i) {
    std::size_t n = std::size_t((translation[i] - _lower[i]) *
                                inverseSpacing);
#ifdef STLIB_DEBUG
    assert(n < std::size_t(1 << _bitsPerTranslation[i]));
#endif
    key <<= _bitsPerTranslation[i];
    key |= n;
  }

  return key;
}


template<typename _Key>
inline
void
SpecialEuclideanCodeUniformGrid<3, _Key>::
decode(Key key, Point* x, Point* y, Point* translation) const
{
  // We unpack in the reverse order that we packed.

  //
  // Unpack the translation.
  //
  for (std::size_t i = D; i-- != 0;) {
    const std::size_t translationMask = ((1 << _bitsPerTranslation[i]) - 1);
    // The center of the cell.
    (*translation)[i] = _lower[i] + ((key & translationMask) + 0.5) *
                        _spacing;
    key >>= _bitsPerTranslation[i];
  }

  //
  // Unpack the rotation.
  //

  const std::size_t extents = 1 << _bitsPerRotation;
  const std::size_t rotationMask = (extents - 1);
  const double rotationSpacing = 2. / (extents - 1);
  // Y axis.
  for (std::size_t i = D; i-- != 0;) {
    (*y)[i] = -1. + (key & rotationMask) * rotationSpacing;
    key >>= _bitsPerRotation;
  }
  // X axis.
  for (std::size_t i = D; i-- != 0;) {
    (*x)[i] = -1. + (key & rotationMask) * rotationSpacing;
    key >>= _bitsPerRotation;
  }
  assert(key == 0);

  //
  // Normalize the axes.
  //
  ext::normalize(x);
  ext::normalize(y);
  // Make y orthogonal to x.
  *y -= ext::dot(*x, *y)** x;
#ifdef STLIB_DEBUG
  assert(numerical::areEqual(ext::dot(*x, *y), 0., 10.));
#endif
  // Re-normalize the rotated y axis.
  ext::normalize(y);
}


} // namespace geom
}
