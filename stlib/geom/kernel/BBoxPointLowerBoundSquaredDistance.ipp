// -*- C++ -*-

#if !defined(__geom_BBoxPointLowerBoundSquaredDistance_ipp__)
#error This file is an implementation detail of BBoxPointLowerBoundSquaredDistance.
#endif

namespace stlib
{
namespace geom
{


template<typename _Float, std::size_t _D>
inline
BBoxPointLowerBoundSquaredDistance<_Float, _D>::
BBoxPointLowerBoundSquaredDistance(BBox const& box) :
  _lower(),
  _upper()
{
  for (std::size_t i = 0; i != _D; ++i) {
    _lower[i] = box.lower[i];
    _upper[i] = box.upper[i];
  }
}


template<typename _Float, std::size_t _D>
inline
_Float
BBoxPointLowerBoundSquaredDistance<_Float, _D>::
operator()(AlignedPoint const& x) const
{
  _Float d2 = 0;
  for (std::size_t i = 0; i != _D; ++i) {
    _Float const d = std::max(_lower[i] - x[i], _Float(0)) +
      std::max(x[i] - _upper[i], _Float(0));
    d2 += d * d;
  }
  return d2;
}


} // namespace geom
} // namespace stlib
