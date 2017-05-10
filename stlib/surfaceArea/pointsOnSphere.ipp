// -*- C++ -*-

#if !defined(__surfaceArea_kernel_pointsOnSphere_ipp__)
#error This file is an implementation detail of pointsOnSphere.
#endif

namespace stlib
{
namespace surfaceArea
{

template<typename _Point, typename _OutputIterator>
inline
void
distributePointsOnSphereWithGoldenSectionSpiral(std::size_t size,
    _OutputIterator points)
{
  // Check the singular case.
  if (size == 0) {
    return;
  }
  const double Delta = numerical::Constants<double>::Pi() *
                       (3. - std::sqrt(5.));
  double longitude = 0;
  double dz = 2.0 / size;
  double z = 1. - dz / 2;
  _Point p;
  double r;
  for (std::size_t i = 0; i != size; ++i) {
    r = std::sqrt(1. - z * z);
    p[0] = r * std::cos(longitude);
    p[1] = r * std::sin(longitude);
    p[2] = z;
    *points++ = p;
    z -= dz;
    longitude += Delta;
  }
}

} // namespace surfaceArea
}
