// -*- C++ -*-

#if !defined(__geom_OrientedBBox_ipp__)
#error This file is an implementation detail of the class OrientedBBox.
#endif

namespace stlib
{
namespace geom
{


template<typename _Float, std::size_t _D>
inline
void
OrientedBBox<_Float, _D>::
buildPca(std::vector<Point> const& points)
{
  assert(! points.empty());

  // Compute the mean. Use this for the initial center.
  center = ext::filled_array<Point>(0);
  for (std::size_t i = 0; i != points.size(); ++i) {
    center += points[i];
  }
  center /= _Float(points.size());

  // Subtract the mean.
  std::vector<Point> centered(points);
  for (std::size_t i = 0; i != centered.size(); ++i) {
    centered[i] -= center;
  }

  // Make a mean-centered matrix.
  typedef Eigen::Matrix<_Float, Eigen::Dynamic, Eigen::Dynamic> Matrix;
  Matrix m(int(points.size()), int(Dimension));
  for (std::size_t row = 0; row != points.size(); ++row) {
    for (std::size_t col = 0; col != Dimension; ++col) {
      m(row, col) = centered[row][col];
    }
  }

  // Compute the SVD.
  Eigen::JacobiSVD<Matrix> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix v = svd.matrixV();
  assert(std::size_t(v.rows()) == Dimension);
  assert(std::size_t(v.cols()) == Dimension);
  // Set the axes.
  for (std::size_t i = 0; i != Dimension; ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      axes[i][j] = v(j, i);
    }
  }

  _updateCenterCalculateRadii(centered);
}


template<typename _Float, std::size_t _D>
inline
void
OrientedBBox<_Float, _D>::
buildPcaRotate(std::vector<Point> const& points)
{
  static_assert(Dimension >= 2, "Only defined for Dimension >= 2.");
  // Build the OBB using PCA.
  buildPca(points);

  // Try a rotation that changes the first and second principal directions.
  OrientedBBox rotated = *this;
  rotated.axes[0] = axes[0] + axes[1];
  ext::normalize(&rotated.axes[0]);
  rotated.axes[1] = axes[0] - axes[1];
  ext::normalize(&rotated.axes[1]);
  std::vector<Point> centered(points);
  for (std::size_t i = 0; i != centered.size(); ++i) {
    centered[i] -= rotated.center;
  }
  rotated._updateCenterCalculateRadii(centered);

  // See if the rotated OBB is better.
  if (ext::product(rotated.radii) < ext::product(radii)) {
    *this = rotated;
  }
}


template<typename _Float, std::size_t _D>
inline
void
OrientedBBox<_Float, _D>::
_updateCenterCalculateRadii(std::vector<Point> const& points)
{
  // Calculate a bounding box with the oriented axes.
  Point lower =
    ext::filled_array<Point>(std::numeric_limits<_Float>::infinity());
  Point upper =
    ext::filled_array<Point>(- std::numeric_limits<_Float>::infinity());
  for (std::size_t i = 0; i != points.size(); ++i) {
    for (std::size_t j = 0; j != Dimension; ++j) {
      _Float const x = ext::dot(points[i], axes[j]);
      if (x < lower[j]) {
        lower[j] = x;
      }
      if (x > upper[j]) {
        upper[j] = x;
      }
    }
  }
  // Re-center and calculate the radii.
  for (std::size_t i = 0; i != Dimension; ++i) {
    center += (_Float(0.5) * (lower[i] + upper[i])) * axes[i];
    radii[i] = _Float(0.5) * (upper[i] - lower[i]);
  }
}


template<typename _Float, std::size_t _D>
template<typename _ForwardIterator>
inline
void
OrientedBBox<_Float, _D>::
buildPca(_ForwardIterator begin, _ForwardIterator end)
{
  // Convert to a vector of points.
  std::vector<Point> points(std::distance(begin, end));
  for (std::size_t i = 0; i != points.size(); ++i) {
    points[i] = ext::ConvertArray<_Float>::convert(*begin++);
  }
  buildPca(points);
}


template<typename _Float, std::size_t _D>
inline
typename OrientedBBox<_Float, _D>::Point
OrientedBBox<_Float, _D>::
transform(Point p) const
{
  p -= center;
  Point t;
  for (std::size_t i = 0; i != Dimension; ++i) {
    t[i] = ext::dot(p, axes[i]);
  }
  return t;
}


template<typename _Float, std::size_t _D>
inline
void
OrientedBBox<_Float, _D>::
transform(std::vector<_Float, simd::allocator<_Float> > const& input,
          std::vector<_Float, simd::allocator<_Float> >* output) const
{
  typedef typename simd::Vector<_Float>::Type Vector;
  std::size_t const VectorSize = simd::Vector<_Float>::Size;

  output->resize(input.size());
  _Float const* in = &input[0];
  _Float* out = &(*output)[0];
  std::array<Vector, Dimension> coords;

  // Cache the center.
  std::array<Vector, Dimension> centerCoords;
  for (std::size_t j = 0; j != Dimension; ++j) {
    centerCoords[j] = simd::set1(center[j]);
  }
  // Cache the axes.
  std::array<std::array<Vector, Dimension>, Dimension> axesCoords;
  for (std::size_t j = 0; j != Dimension; ++j) {
    for (std::size_t k = 0; k != Dimension; ++k) {
      axesCoords[j][k] = simd::set1(axes[j][k]);
    }
  }

  // For each block of points.
  std::size_t numBlocks = input.size() / (Dimension * VectorSize);
  for (std::size_t i = 0; i != numBlocks; ++i) {
    // Subtract the center.
    for (std::size_t j = 0; j != Dimension; ++j) {
      coords[j] = simd::load(in) - centerCoords[j];
      in += VectorSize;
    }
    for (std::size_t j = 0; j != Dimension; ++j) {
      Vector r = simd::setzero<_Float>();
      for (std::size_t k = 0; k != Dimension; ++k) {
        r += coords[k] * axesCoords[j][k];
      }
      simd::store(out, r);
      out += VectorSize;
    }
  }
}


} // namespace geom
}
