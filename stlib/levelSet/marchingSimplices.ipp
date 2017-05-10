// -*- C++ -*-

#if !defined(__levelSet_marchingSimplices_ipp__)
#error This file is an implementation detail of marchingSimplices.
#endif

namespace stlib
{
namespace levelSet
{


// http://fluxionsdividebyzero.com/p1/math/calculus/geometry/g017.html


//
// Simplex content.
//


//! Return the content and boundary of the portion of the triangle defined by the level set.
/*! \pre The values must be of mixed signs. */
template<typename _T, typename _OutputIterator>
inline
void
contentAndBoundary(std::array<_T, 3> values,
                   std::array<std::array<_T, 2>, 3> locations,
                   const std::array<_T, 2>& lowerCorner,
                   const _T simplexContent, _T* content, _T* boundary,
                   _OutputIterator vertices)
{
  typedef std::array<_T, 2> Point;

  // Order so that the first vertex is negative.
  if (values[0] >= 0) {
    if (values[1] < 0) {
      // 0 1 2 -> 1 0 2
      std::swap(values[0], values[1]);
      std::swap(locations[0], locations[1]);
    }
    else {
      // 0 1 2 -> 2 1 0
      std::swap(values[0], values[2]);
      std::swap(locations[0], locations[2]);
    }
  }
#ifdef STLIB_DEBUG
  assert(values[0] < 0);
#endif
  if (values[1] >= 0 && values[2] < 0) {
    std::swap(values[1], values[2]);
    std::swap(locations[1], locations[2]);
  }

  // There are two possible patterns:
  // --+ -++
  Point a, b;
  if (values[1] < 0) {
    //  +
    // --
    a = locations[2] + (- values[2] / (values[0] - values[2])) *
        (locations[0] - locations[2]);
    b = locations[2] + (- values[2] / (values[1] - values[2])) *
        (locations[1] - locations[2]);
    // Use the area of the simplex to only compute one triangle area.
    *content = simplexContent -
               std::abs(geom::computeArea(locations[0], a, b));
  }
  else {
    //  +
    // -+
    a = locations[0] + (- values[0] / (values[1] - values[0])) *
        (locations[1] - locations[0]);
    b = locations[0] + (- values[0] / (values[2] - values[0])) *
        (locations[2] - locations[0]);
    *content = std::abs(geom::computeArea(locations[0], a, b));
  }
  *boundary = ext::euclideanDistance(a, b);
  // Translate the point only after computing the content in order to
  // minimize the round-off errors.
  a += lowerCorner;
  b += lowerCorner;
  *vertices++ = a;
  *vertices++ = b;
}


//! Return the content of the portion of the triangle defined by the level set.
/*! \pre The values must be of mixed signs. */
template<typename _T>
inline
_T
content(std::array<_T, 3> values,
        std::array<std::array<_T, 2>, 3> locations,
        const _T simplexContent)
{
  const std::array<_T, 2> lowerCorner = {{0, 0}};
  _T c, b;
  ads::TrivialOutputIterator vertices;
  contentAndBoundary(values, locations, lowerCorner, simplexContent, &c, &b,
                     vertices);
  return c;
}


//! Return the content of the portion of the triangle defined by the level set.
/*! \pre The values must be of mixed signs. */
template<typename _T>
inline
_T
boundary(std::array<_T, 3> values,
         std::array<std::array<_T, 2>, 3> locations,
         const _T simplexContent)
{
  const std::array<_T, 2> lowerCorner = {{0, 0}};
  _T c, b;
  ads::TrivialOutputIterator vertices;
  contentAndBoundary(values, locations, lowerCorner, simplexContent, &c, &b,
                     vertices);
  return b;
}


//! Return the content and boundary of the portion of the tetrahedron defined by the level set.
/*! \pre The values must be of mixed signs. */
template<typename _T, typename _OutputIterator>
inline
void
contentAndBoundary(std::array<_T, 4> values,
                   std::array<std::array<_T, 3>, 4> locations,
                   const std::array<_T, 3>& lowerCorner,
                   const _T simplexContent, _T* content, _T* boundary,
                   _OutputIterator vertices)
{
  typedef std::array<_T, 3> Point;

  // Order so that negative values precede positive ones.
  ads::sortTogether(values.begin(), values.end(), locations.begin(),
                    locations.end());
#ifdef STLIB_DEBUG
  assert(values[0] < 0 && values[3] > 0);
#endif

  // There are three possible patterns:
  // ---+ --++ -+++
  if (values[1] > 0) {
    // ++
    // -+
    std::array<Point, 3> pos;
    for (std::size_t i = 0; i != pos.size(); ++i) {
      pos[i] = locations[0] + (- values[0] / (values[i + 1] - values[0])) *
               (locations[i + 1] - locations[0]);
    }
    *content = std::abs(geom::computeVolume(locations[0], pos[0], pos[1],
                                            pos[2]));
    *boundary = geom::computeArea(pos);
    // Translate the points only after computing the content in order to
    // minimize the round-off errors.
    pos[0] += lowerCorner;
    pos[1] += lowerCorner;
    pos[2] += lowerCorner;
    *vertices++ = pos[0];
    *vertices++ = pos[1];
    *vertices++ = pos[2];
  }
  else if (values[2] > 0) {
    // ++
    // --
    // Let a and b be the two points with negative values.
    // Let a0 and a1 be the intersection points between a and the two
    // points with positive values. Likewise for b.
    // The volume is composed of three tetrahedra:
    // (a, a0, a1, b1)
    // (b, b0, b1, a0)
    // (a, a0, b, b1)
    std::array<Point, 2> a;
    for (std::size_t i = 0; i != 2; ++i) {
      a[i] = locations[0] + (- values[0] / (values[i + 2] - values[0])) *
             (locations[i + 2] - locations[0]);
    }
    std::array<Point, 2> b;
    for (std::size_t i = 0; i != 2; ++i) {
      b[i] = locations[1] + (- values[1] / (values[i + 2] - values[1])) *
             (locations[i + 2] - locations[1]);
    }
    *content = std::abs(geom::computeVolume(locations[0], a[0], a[1], b[1])) +
               std::abs(geom::computeVolume(locations[1], b[0], b[1], a[0])) +
               std::abs(geom::computeVolume(locations[0], a[0], locations[1], b[1]));
    *boundary = geom::computeArea(a[0], a[1], b[1]) +
                geom::computeArea(b[0], b[1], a[0]);
    // Translate the points only after computing the content in order to
    // minimize the round-off errors.
    a[0] += lowerCorner;
    a[1] += lowerCorner;
    b[0] += lowerCorner;
    b[1] += lowerCorner;
    *vertices++ = a[0];
    *vertices++ = a[1];
    *vertices++ = b[1];
    *vertices++ = b[0];
    *vertices++ = b[1];
    *vertices++ = a[0];
  }
  else {
    // -+
    // --
    std::array<Point, 3> pos;
    for (std::size_t i = 0; i != pos.size(); ++i) {
      pos[i] = locations[3] + (- values[3] / (values[i] - values[3])) *
               (locations[i] - locations[3]);
    }
    // Subtract the positive region from the simplex.
    *content = simplexContent -
               std::abs(geom::computeVolume(locations[3], pos[0], pos[1], pos[2]));
    *boundary = geom::computeArea(pos);
    // Translate the points only after computing the content in order to
    // minimize the round-off errors.
    pos[0] += lowerCorner;
    pos[1] += lowerCorner;
    pos[2] += lowerCorner;
    *vertices++ = pos[0];
    *vertices++ = pos[1];
    *vertices++ = pos[2];
  }
}


//! Return the content of the portion of the tetrahedron defined by the level set.
/*! \pre The values must be of mixed signs. */
template<typename _T>
inline
_T
content(std::array<_T, 4> values,
        std::array<std::array<_T, 3>, 4> locations,
        const _T simplexContent)
{
  const std::array<_T, 3> lowerCorner = {{0, 0, 0}};
  _T c, b;
  ads::TrivialOutputIterator vertices;
  contentAndBoundary(values, locations, lowerCorner, simplexContent,
                     &c, &b, vertices);
  return c;
}


//! Return the boundary of the portion of the tetrahedron defined by the level set.
/*! \pre The values must be of mixed signs. */
template<typename _T>
inline
_T
boundary(std::array<_T, 4> values,
         std::array<std::array<_T, 3>, 4> locations,
         const _T simplexContent)
{
  const std::array<_T, 3> lowerCorner = {{0, 0, 0}};
  _T c, b;
  ads::TrivialOutputIterator vertices;
  contentAndBoundary(values, locations, lowerCorner, simplexContent,
                     &c, &b, vertices);
  return b;
}


//
// Voxel content. The name starts with "voxel" so these functions are not
// confused with the patch versions when the patch extent is 2.
//


//! Return the content and boundary of the object defined by the level set.
/*! \pre The grid values must have mixed signs. */
template<typename _T, typename _OutputIterator>
inline
void
voxelContentAndBoundary(container::EquilateralArray<_T, 1, 2> voxel,
                        const std::array<_T, 1>& lowerCorner,
                        const _T spacing, _T* content, _T* boundary,
                        _OutputIterator vertices)
{
  // Order the vertices.
  if (voxel[0] > 0) {
    std::swap(voxel[0], voxel[1]);
  }
#ifdef STLIB_DEBUG
  assert(voxel[0] < 0 && voxel[1] > 0);
#endif
  // a + (b - a) t / spacing == 0
  // t == -a * spacing / (b - a)
  // content = t
  const _T t = - voxel[0] * spacing / (voxel[1] - voxel[0]);
  *content = t;
  // With this convention, a single component has unit boundary.
  *boundary = 0.5;
  *vertices++ = lowerCorner + std::array<_T, 1>{{t}};
}


//! Return the content of the object defined by the level set.
/*! \pre The grid values must have mixed signs. */
template<typename _T>
inline
_T
voxelContent(container::EquilateralArray<_T, 1, 2> voxel, const _T spacing)
{
  _T c, b;
  ads::TrivialOutputIterator vertices;
  voxelContentAndBoundary(voxel, ext::filled_array<std::array<_T, 1> >(0),
                          spacing, &c, &b, vertices);
  return c;
}


//! Return the boundary of the object defined by the level set.
/*! \pre The grid values must have mixed signs. */
template<typename _T>
inline
_T
voxelBoundary(container::EquilateralArray<_T, 1, 2> voxel, const _T spacing)
{
  _T c, b;
  ads::TrivialOutputIterator vertices;
  voxelContentAndBoundary(voxel, ext::filled_array<std::array<_T, 1> >(0),
                          spacing, &c, &b, vertices);
  return b;
}


// CONTINUE: Make this function a class.
//! Return the content of the object defined by the level set.
/*! \pre The grid values must have mixed signs. */
template<typename _T, typename _OutputIterator>
inline
void
voxelContentAndBoundary(const container::EquilateralArray<_T, 2, 2>& voxel,
                        const std::array<_T, 2>& lowerCorner,
                        const _T spacing, _T* content, _T* boundary,
                        _OutputIterator vertices)
{
  typedef std::array<_T, 2> Point;

  const std::array<Point, 4> VertexData = {{
      {{0, 0}},
      {{spacing, 0}},
      {{0, spacing}},
      {{spacing, spacing}}
    }
  };
  const container::EquilateralArray<Point, 2, 2> Vertices(VertexData);
  const _T simplexContent = 0.5 * spacing * spacing;

  *content = 0;
  *boundary = 0;
  _T c, b;
  std::array<_T, 3> values;
  std::array<Point, 3> locations;

  // First triangle.
  values[0] = voxel[0];
  values[1] = voxel[1];
  values[2] = voxel[3];
  if (allNonPositive(values.begin(), values.end())) {
    *content += simplexContent;
  }
  else if (mixedSigns(values.begin(), values.end())) {
    locations[0] = Vertices[0];
    locations[1] = Vertices[1];
    locations[2] = Vertices[3];
    contentAndBoundary(values, locations, lowerCorner, simplexContent, &c,
                       &b, vertices);
    *content += c;
    *boundary += b;
  }

  // Second triangle.
  values[1] = voxel[2];
  if (allNonPositive(values.begin(), values.end())) {
    *content += simplexContent;
  }
  else if (mixedSigns(values.begin(), values.end())) {
    locations[0] = Vertices[0];
    locations[1] = Vertices[2];
    locations[2] = Vertices[3];
    contentAndBoundary(values, locations, lowerCorner, simplexContent, &c,
                       &b, vertices);
    *content += c;
    *boundary += b;
  }
}


//! Return the content of the object defined by the level set.
/*! \pre The grid values must have mixed signs. */
template<typename _T>
inline
_T
voxelContent(const container::EquilateralArray<_T, 2, 2>& voxel,
             const _T spacing)
{
  _T c, b;
  ads::TrivialOutputIterator vertices;
  voxelContentAndBoundary(voxel, ext::filled_array<std::array<_T, 2> >(0),
                          spacing, &c, &b, vertices);
  return c;
}


//! Return the content of the object defined by the level set.
/*! \pre The grid values must have mixed signs. */
template<typename _T>
inline
_T
voxelBoundary(const container::EquilateralArray<_T, 2, 2>& voxel,
              const _T spacing)
{
  _T c, b;
  ads::TrivialOutputIterator vertices;
  voxelContentAndBoundary(voxel, ext::filled_array<std::array<_T, 2> >(0),
                          spacing, &c, &b, vertices);
  return b;
}


// CONTINUE: Make this function a class.
//! Return the content of the object defined by the level set.
/*! \pre The grid values must have mixed signs. */
template<typename _T, typename _OutputIterator>
inline
void
voxelContentAndBoundary(const container::EquilateralArray<_T, 3, 2>& voxel,
                        const std::array<_T, 3>& lowerCorner,
                        const _T spacing, _T* content, _T* boundary,
                        _OutputIterator vertices)
{
  typedef std::array<_T, 3> Point;

  const std::array<Point, 8> VertexData = {{
      {{0, 0, 0}},
      {{spacing, 0, 0}},
      {{0, spacing, 0}},
      {{spacing, spacing, 0}},
      {{0, 0, spacing}},
      {{spacing, 0, spacing}},
      {{0, spacing, spacing}},
      {{spacing, spacing, spacing}}
    }
  };
  const container::EquilateralArray<Point, 3, 2> Vertices(VertexData);
  const _T simplexContent = (1. / 6) * spacing * spacing * spacing;

  // The indices for the six tetrahedra in the diagonal decomposition of
  // the voxel.
  // http://fluxionsdividebyzero.com/p1/math/calculus/geometry/diagg01706.png
  const std::array<std::array<std::size_t, 4>, 6> n = {{
      {{0, 1, 3, 7}},
      {{0, 1, 5, 7}},
      {{0, 2, 3, 7}},
      {{0, 2, 6, 7}},
      {{0, 4, 5, 7}},
      {{0, 4, 6, 7}}
    }
  };

  *content = 0;
  *boundary = 0;
  _T c, b;
  std::array<_T, 4> values;
  std::array<Point, 4> locations;

  // For each tetrahedron.
  for (std::size_t i = 0; i != n.size(); ++i) {
    for (std::size_t j = 0; j != values.size(); ++j) {
      values[j] = voxel[n[i][j]];
      locations[j] = Vertices[n[i][j]];
    }
    if (allNonPositive(values.begin(), values.end())) {
      *content += simplexContent;
    }
    else if (mixedSigns(values.begin(), values.end())) {
      contentAndBoundary(values, locations, lowerCorner, simplexContent,
                         &c, &b, vertices);
#if 0
      // CONTINUE REMOVE
      // This test will currently fail when using float.
      if (!(c >= 0 && c <= simplexContent * 1.000001)) {
        std::cerr << c << ' ' << simplexContent << ' '
                  << geom::computeContent(locations) << '\n'
                  << locations << '\n';
      }
#endif
      *content += c;
      *boundary += b;
    }
  }
}


//! Return the content of the object defined by the level set.
/*! \pre The grid values must have mixed signs. */
template<typename _T>
inline
_T
voxelContent(const container::EquilateralArray<_T, 3, 2>& voxel,
             const _T spacing)
{
  _T c, b;
  ads::TrivialOutputIterator vertices;
  voxelContentAndBoundary(voxel, ext::filled_array<std::array<_T, 3> >(0),
                          spacing, &c, &b, vertices);
  return c;
}


//! Return the boundary of the object defined by the level set.
/*! \pre The grid values must have mixed signs. */
template<typename _T>
inline
_T
voxelBoundary(const container::EquilateralArray<_T, 3, 2>& voxel,
              const _T spacing)
{
  _T c, b;
  ads::TrivialOutputIterator vertices;
  voxelContentAndBoundary(voxel, ext::filled_array<std::array<_T, 3> >(0),
                          spacing, &c, &b, vertices);
  return b;
}


//
// Extract voxels.
//


template<typename _T, std::size_t N>
inline
void
getVoxel(const container::EquilateralArray<_T, 1, N>& patch,
         const typename container::EquilateralArray<_T, 1, N>::IndexList& index,
         container::EquilateralArray<_T, 1, 2>* voxel)
{
#ifdef STLIB_DEBUG
  assert(index[0] + 1 < patch.extents()[0]);
#endif
  typename container::EquilateralArray<_T, 1, N>::Index i = index[0];
  (*voxel)[0] = patch(i);
  (*voxel)[1] = patch(i + 1);
}


template<typename _T, std::size_t N>
inline
void
getVoxel(const container::EquilateralArray<_T, 2, N>& patch,
         const typename container::EquilateralArray<_T, 2, N>::IndexList& index,
         container::EquilateralArray<_T, 2, 2>* voxel)
{
#ifdef STLIB_DEBUG
  assert(index[0] + 1 < patch.extents()[0] &&
         index[1] + 1 < patch.extents()[1]);
#endif
  typename container::EquilateralArray<_T, 2, N>::Index
  i = index[0], j = index[1];
  (*voxel)[0] = patch(i, j);
  (*voxel)[1] = patch(i + 1, j);
  (*voxel)[2] = patch(i, j + 1);
  (*voxel)[3] = patch(i + 1, j + 1);
}


template<typename _T, std::size_t N>
inline
void
getVoxel(const container::EquilateralArray<_T, 3, N>& patch,
         const typename container::EquilateralArray<_T, 3, N>::IndexList& index,
         container::EquilateralArray<_T, 3, 2>* voxel)
{
#ifdef STLIB_DEBUG
  assert(index[0] + 1 < patch.extents()[0] &&
         index[1] + 1 < patch.extents()[1] &&
         index[2] + 1 < patch.extents()[2]);
#endif
  typename container::EquilateralArray<_T, 3, N>::Index
  i = index[0], j = index[1], k = index[2];
  (*voxel)[0] = patch(i, j, k);
  (*voxel)[1] = patch(i + 1, j, k);
  (*voxel)[2] = patch(i, j + 1, k);
  (*voxel)[3] = patch(i + 1, j + 1, k);
  (*voxel)[4] = patch(i, j, k + 1);
  (*voxel)[5] = patch(i + 1, j, k + 1);
  (*voxel)[6] = patch(i, j + 1, k + 1);
  (*voxel)[7] = patch(i + 1, j + 1, k + 1);
}


//
// Patch content.
//

// Return the content of the object defined by the level set.
template<typename _T, std::size_t _D, std::size_t N, typename _OutputIterator>
inline
void
contentAndBoundary(const container::EquilateralArray<_T, _D, N>& patch,
                   const std::array<_T, _D>& lowerCorner,
                   const _T spacing, _T* content, _T* boundary,
                   _OutputIterator vertices)
{
  typedef container::EquilateralArray<_T, _D, 2> Voxel;
  typedef typename Voxel::IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // The content of voxel.
  _T voxelContent = 1;
  for (std::size_t i = 0; i != _D; ++i) {
    voxelContent *= spacing;
  }

  *content = 0;
  *boundary = 0;
  _T c, b;
  Voxel voxel;
  const std::array<_T, _D> delta =
    ext::filled_array<std::array<_T, _D> >(spacing);
  // Note the the voxel extents are one less than the vertex extents.
  const IndexList extents = patch.extents() - std::size_t(1);
  // Loop over the voxels.
  const Iterator end = Iterator::end(extents);
  for (Iterator i = Iterator::begin(extents); i != end; ++i) {
    // Extract the voxel of function values.
    getVoxel(patch, *i, &voxel);
    if (allNonPositive(voxel.begin(), voxel.end())) {
      *content += voxelContent;
    }
    else if (mixedSigns(voxel.begin(), voxel.end())) {
      voxelContentAndBoundary
        (voxel, lowerCorner + stlib::ext::convert_array<_T>(*i) * delta,
         spacing, &c, &b, vertices);
      *content += c;
      *boundary += b;
    }
  }
}


//! Return the content of the object defined by the level set.
template<typename _T, std::size_t _D, std::size_t N>
inline
_T
content(const container::EquilateralArray<_T, _D, N>& patch,
        const _T spacing)
{
  _T c, b;
  ads::TrivialOutputIterator vertices;
  contentAndBoundary(patch, ext::filled_array<std::array<_T, _D> >(0),
                     spacing, &c, &b, vertices);
  return c;
}


//! Return the boundary of the object defined by the level set.
template<typename _T, std::size_t _D, std::size_t N>
inline
_T
boundary(const container::EquilateralArray<_T, _D, N>& patch,
         const _T spacing)
{
  _T c, b;
  ads::TrivialOutputIterator vertices;
  contentAndBoundary(patch, ext::filled_array<std::array<_T, _D> >(0),
                     spacing, &c, &b, vertices);
  return b;
}


// Accumulate the content and boundary of each component of the object.
template<typename _T, std::size_t _D, std::size_t N, typename _I,
         typename _OutputIterator>
inline
void
contentAndBoundary(const container::EquilateralArray<_T, _D, N>& functionPatch,
                   const container::EquilateralArray<_I, _D, N>& componentsPatch,
                   const std::array<_T, _D>& lowerCorner,
                   const _T spacing, std::vector<_T>* content,
                   std::vector<_T>* boundary, _OutputIterator vertices)
{
  typedef container::EquilateralArray<_T, _D, 2> FunctionVoxel;
  typedef container::EquilateralArray<_I, _D, 2> ComponentsVoxel;
  typedef typename FunctionVoxel::IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // The content of voxel.
  _T voxelContent = 1;
  for (std::size_t i = 0; i != _D; ++i) {
    voxelContent *= spacing;
  }

  _T c, b;
  FunctionVoxel functionVoxel;
  ComponentsVoxel componentsVoxel;
  const std::array<_T, _D> delta =
    ext::filled_array<std::array<_T, _D> >(spacing);
  // Note the the voxel extents are one less than the vertex extents.
  const IndexList extents = functionPatch.extents() - std::size_t(1);
  // Loop over the voxels.
  const Iterator end = Iterator::end(extents);
  for (Iterator i = Iterator::begin(extents); i != end; ++i) {
    // Extract the voxel of function values.
    getVoxel(functionPatch, *i, &functionVoxel);
    getVoxel(componentsPatch, *i, &componentsVoxel);
    if (allNonPositive(functionVoxel.begin(), functionVoxel.end())) {
      const _I component = ext::min(componentsVoxel);
      (*content)[component] += voxelContent;
    }
    else if (mixedSigns(functionVoxel.begin(), functionVoxel.end())) {
      voxelContentAndBoundary
        (functionVoxel, lowerCorner + stlib::ext::convert_array<_T>(*i) * delta,
         spacing, &c, &b, vertices);
      const _I component = ext::min(componentsVoxel);
      (*content)[component] += c;
      (*boundary)[component] += b;
    }
  }
}


//
// Grid content.
//


// Return the content and boundary of the object defined by the level set.
template<typename _T, std::size_t _D, std::size_t N, typename _OutputIterator>
inline
void
contentAndBoundary(const Grid<_T, _D, N>& grid, _T* content, _T* boundary,
                   _OutputIterator vertices)
{
  typedef typename Grid<_T, _D, N>::VoxelPatch VoxelPatch;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // The content of an entire patch of voxels.
  _T patchContent = 1;
  const _T patchLength = grid.getVoxelPatchLength();
  for (std::size_t i = 0; i != _D; ++i) {
    patchContent *= patchLength;
  }

  VoxelPatch patch;
  *content = 0;
  *boundary = 0;
  _T c, b;
  // Loop over the patches.
  const Iterator end = Iterator::end(grid.extents());
  for (Iterator i = Iterator::begin(grid.extents()); i != end; ++i) {
    if (grid(*i).isRefined()) {
      grid.getVoxelPatch(*i, &patch);
      contentAndBoundary(patch, grid.getPatchLowerCorner(*i), grid.spacing,
                         &c, &b, vertices);
      *content += c;
      *boundary += b;
    }
    // If the whole patch is inside the object.
    else if (grid(*i).fillValue < 0) {
      *content += patchContent;
    }
  }
}


// Return the content and boundary of each component of the object.
/* The sizes of the content and boundary vectors will be set to the number
   of components. */
template<typename _T, std::size_t _D, std::size_t N, typename _OutputIterator>
inline
void
contentAndBoundary(const Grid<_T, _D, N>& grid,
                   std::vector<_T>* content, std::vector<_T>* boundary,
                   _OutputIterator vertices)
{
  typedef typename Grid<_T, _D, N>::VoxelPatch FunctionPatch;
  typedef typename Grid<unsigned, _D, N, _T>::VoxelPatch ComponentsPatch;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  Grid<unsigned, _D, N, _T> components(grid.domain(), grid.extents());
  const unsigned numComponents = labelComponents(grid, &components);
  content->resize(numComponents);
  boundary->resize(numComponents);
  // If there are no components, there is no object.
  if (numComponents == 0) {
    return;
  }

  // The content of an entire patch of voxels.
  _T patchContent = 1;
  const _T patchLength = grid.getVoxelPatchLength();
  for (std::size_t i = 0; i != _D; ++i) {
    patchContent *= patchLength;
  }

  FunctionPatch functionPatch;
  ComponentsPatch componentsPatch;
  std::fill(content->begin(), content->end(), _T(0));
  std::fill(boundary->begin(), boundary->end(), _T(0));
  // Loop over the patches.
  const Iterator end = Iterator::end(grid.extents());
  for (Iterator i = Iterator::begin(grid.extents()); i != end; ++i) {
    if (grid(*i).isRefined()) {
      grid.getVoxelPatch(*i, &functionPatch);
      components.getVoxelPatch(*i, &componentsPatch);
      contentAndBoundary(functionPatch, componentsPatch,
                         grid.getPatchLowerCorner(*i), grid.spacing,
                         content, boundary, vertices);
    }
    // If the whole patch is inside the object.
    else if (grid(*i).fillValue < 0) {
      (*content)[components(*i).fillValue] += patchContent;
    }
  }
}


} // namespace levelSet
}
