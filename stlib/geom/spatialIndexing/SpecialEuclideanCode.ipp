// -*- C++ -*-

#if !defined(__geom_spatialIndexing_SpecialEuclideanCode_ipp__)
#error This file is an implementation detail of the class SpecialEuclideanCode.
#endif

namespace stlib
{
namespace geom
{


template<std::size_t _SubdivisionLevels, typename _Key>
inline
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
SpecialEuclideanCode(const BBox& domain, const double spacing) :
  _lower(),
  _spacing(spacing),
  _bitsPerTranslation(),
  _centroids()
{
  static_assert(_SubdivisionLevels > 0, "Bad number of subdivisions.");

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
  // Check that the code can be packed into the integer type for the key.
  assert(2 * BitsPerAxis + ext::sum(_bitsPerTranslation) <=
         std::size_t(std::numeric_limits<_Key>::digits));
  // From the number of bits, determine the cell extents.
  std::array<std::size_t, D> extents;
  for (std::size_t i = 0; i != D; ++i) {
    extents[i] = 1 << _bitsPerTranslation[i];
  }
  // Determine the lower bound.
  _lower = 0.5 * (domain.lower + domain.upper) -
    (0.5 * spacing) * ext::convert_array<double>(extents);

  // Compute the centroids of the spherical triangles at each level of
  // subdivision.
  computeCentroids();
  // Compute the adjacencies at the finest level of subdivision.
  computeAdjacencies();
}


template<std::size_t _SubdivisionLevels, typename _Key>
inline
void
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
computeCentroids()
{
  // Three points form a triangle.
  typedef std::array<Point, D> Triangle;

  std::array<Point, 6> axes = {{
      {{ -1, 0, 0}}, // -x
      {{1, 0, 0}},  // +x
      {{0, -1, 0}}, // -y
      {{0, 1, 0}},  // +y
      {{0, 0, -1}}, // -z
      {{0, 0, 1}}   // +z
    }
  };
  // Start by computing the corners of the triangles.
  std::array < std::vector<Triangle>, _SubdivisionLevels + 1 > corners;

  // The first level. 8 octants. 1 triangle per octant. We need to order the
  // vertices so that the numbering matches up. For two adjacent triangles,
  // the shared vertices have the same local index and hence the mirror
  // vertices also have the same local index.
  corners[0].resize(8);
  // -x -y -z
  corners[0][0][0] = axes[0];
  corners[0][0][1] = axes[2];
  corners[0][0][2] = axes[4];
  // +x -y -z
  corners[0][1][0] = axes[1];
  corners[0][1][1] = axes[2];
  corners[0][1][2] = axes[4];
  // -x +y -z
  corners[0][2][0] = axes[0];
  corners[0][2][1] = axes[3];
  corners[0][2][2] = axes[4];
  // +x +y -z
  corners[0][3][0] = axes[1];
  corners[0][3][1] = axes[3];
  corners[0][3][2] = axes[4];
  // -x -y +z
  corners[0][4][0] = axes[0];
  corners[0][4][1] = axes[2];
  corners[0][4][2] = axes[5];
  // +x -y +z
  corners[0][5][0] = axes[1];
  corners[0][5][1] = axes[2];
  corners[0][5][2] = axes[5];
  // -x +y +z
  corners[0][6][0] = axes[0];
  corners[0][6][1] = axes[3];
  corners[0][6][2] = axes[5];
  // +x +y +z
  corners[0][7][0] = axes[1];
  corners[0][7][1] = axes[3];
  corners[0][7][2] = axes[5];

  /*  2
     / \
    1---0
   / \ / \
  0---2---1 */
  // Calculate the corners for the subdivided meshes.
  std::array<Point, 3> midpoints;
  for (std::size_t i = 0; i != _SubdivisionLevels; ++i) {
    corners[i + 1].resize(corners[i].size() * 4);
    for (std::size_t j = 0; j != corners[i].size(); ++j) {
      const Triangle& t = corners[i][j];
      // The midpoints of the edges.
      for (std::size_t k = 0; k != 3; ++k) {
        midpoints[k] = 0.5 * (t[(k + 1) % 3] + t[(k + 2) % 3]);
        ext::normalize(&midpoints[k]);
      }
      // The four subdivided triangles.
      {
        Triangle& s = corners[i + 1][4 * j + 0];
        s[0] = t[0];
        s[1] = midpoints[1];
        s[2] = midpoints[2];
      }
      {
        Triangle& s = corners[i + 1][4 * j + 1];
        s[0] = midpoints[0];
        s[1] = t[1];
        s[2] = midpoints[2];
      }
      {
        Triangle& s = corners[i + 1][4 * j + 2];
        s[0] = midpoints[0];
        s[1] = midpoints[1];
        s[2] = t[2];
      }
      {
        Triangle& s = corners[i + 1][4 * j + 3];
        s[0] = midpoints[0];
        s[1] = midpoints[1];
        s[2] = midpoints[2];
      }
    }
  }

  // Finally, compute the centroids from the triangles.
  for (std::size_t i = 0; i != _centroids.size(); ++i) {
    _centroids[i].resize(corners[i + 1].size());
    for (std::size_t j = 0; j != _centroids[i].size(); ++j) {
      const Triangle& t = corners[i + 1][j];
      _centroids[i][j] = (1. / 3) * (t[0] + t[1] + t[2]);
      ext::normalize(&_centroids[i][j]);
    }
  }
}


template<std::size_t _SubdivisionLevels, typename _Key>
inline
void
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
computeAdjacencies()
{
  // Start by computing the corners of the triangles.
  std::array < std::vector<AdjacentList>, _SubdivisionLevels + 1 > adj;

  adj[0].resize(8);
  // 1 2 4
  adj[0][0][0] = 1;
  adj[0][0][1] = 2;
  adj[0][0][2] = 4;
  // 0 3 5
  adj[0][1][0] = 0;
  adj[0][1][1] = 3;
  adj[0][1][2] = 5;
  // 3 0 6
  adj[0][2][0] = 3;
  adj[0][2][1] = 0;
  adj[0][2][2] = 6;
  // 2 1 7
  adj[0][3][0] = 2;
  adj[0][3][1] = 1;
  adj[0][3][2] = 7;
  // 5 6 0
  adj[0][4][0] = 5;
  adj[0][4][1] = 6;
  adj[0][4][2] = 0;
  // 4 7 1
  adj[0][5][0] = 4;
  adj[0][5][1] = 7;
  adj[0][5][2] = 1;
  // 7 4 2
  adj[0][6][0] = 7;
  adj[0][6][1] = 4;
  adj[0][6][2] = 2;
  // 6 5 3
  adj[0][7][0] = 6;
  adj[0][7][1] = 5;
  adj[0][7][2] = 3;

  // Define the adjacencies for the refined meshes.
  for (std::size_t i = 0; i != _SubdivisionLevels; ++i) {
    adj[i + 1].resize(adj[i].size() * 4);
    for (std::size_t j = 0; j != adj[i].size(); ++j) {
      const AdjacentList& a = adj[i][j];
      {
        AdjacentList& r = adj[i + 1][4 * j + 0];
        r[0] = 4 * j + 3;
        r[1] = 4 * a[2] + 0;
        r[2] = 4 * a[1] + 0;
      }
      {
        AdjacentList& r = adj[i + 1][4 * j + 1];
        r[0] = 4 * a[2] + 1;
        r[1] = 4 * j + 3;
        r[2] = 4 * a[0] + 1;
      }
      {
        AdjacentList& r = adj[i + 1][4 * j + 2];
        r[0] = 4 * a[1] + 2;
        r[1] = 4 * a[0] + 2;
        r[2] = 4 * j + 3;
      }
      {
        AdjacentList& r = adj[i + 1][4 * j + 3];
        r[0] = 4 * j + 0;
        r[1] = 4 * j + 1;
        r[2] = 4 * j + 2;
      }
    }
  }

#ifdef STLIB_DEBUG
  for (std::size_t i = 0; i != adj.size(); ++i) {
    for (std::size_t j = 0; j != adj[i].size(); ++j) {
      for (std::size_t k = 0; k != 3; ++k) {
        assert(adj[i][adj[i][j][k]][k] == j);
      }
    }
  }
#endif

  // Record the adjacencies on the final level of subdivision.
  _adjacencies = adj[_SubdivisionLevels];
}


template<std::size_t _SubdivisionLevels, typename _Key>
inline
typename SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::Key
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
encode(const Quaternion& q, const Point& translation) const
{
#ifdef STLIB_DEBUG
  // Check that q is a unit quaternion.
  assert(numerical::areEqual(norm(q), 1., 100.));
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


template<std::size_t _SubdivisionLevels, typename _Key>
inline
void
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
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
                    // The formula in "Geometric Tools for Computer Graphics"
                    // has the wrong sign.
                    //(ax[0][2] - ax[2][0]) * fac,
                    (ax[2][0] - ax[0][2]) * fac,
                    (ax[0][1] - ax[1][0]) * fac);
  }
  else {
    // Pick the largest component of the axis of rotation.
    if (ax[0][0] > ax[1][1] && ax[0][0] > ax[2][2]) {
      const double x = 0.5 * std::sqrt(ax[0][0] - ax[1][1] - ax[2][2] + 1);
      const double fac = 1. / (4 * x);
      *q = Quaternion(//(ax[2][1] - ax[1][2]) * fac,
             (ax[1][2] - ax[2][1]) * fac,
             x,
             (ax[1][0] + ax[0][1]) * fac,
             (ax[2][0] + ax[0][2]) * fac);
    }
    else if (ax[1][1] > ax[0][0] && ax[1][1] > ax[2][2]) {
      const double y = 0.5 * std::sqrt(ax[1][1] - ax[0][0] - ax[2][2] + 1);
      const double fac = 1. / (4 * y);
      *q = Quaternion(//(ax[0][2] - ax[2][0]) * fac,
             (ax[2][0] - ax[0][2]) * fac,
             (ax[1][0] + ax[0][1]) * fac,
             y,
             (ax[2][1] + ax[1][2]) * fac);
    }
    else {
      const double z = 0.5 * std::sqrt(ax[2][2] - ax[0][0] - ax[1][1] + 1);
      const double fac = 1. / (4 * z);
      // The formula in "Geometric Tools for Computer Graphics"
      // has the wrong sign.
      *q = Quaternion(//(ax[1][0] - ax[0][1]) * fac,
             (ax[0][1] - ax[1][0]) * fac,
             (ax[2][0] + ax[0][2]) * fac,
             (ax[2][1] + ax[1][2]) * fac,
             z);
    }
  }
}


template<std::size_t _SubdivisionLevels, typename _Key>
inline
typename SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::Key
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
encode(const Point& x, const Point& y, const Point& translation) const
{
  //
  // First pack the rotation.
  //
  Key key = 0;
  key <<= BitsPerAxis;
  key |= encode(x);
  key <<= BitsPerAxis;
  key |= encode(y);
  //
  // Then pack the translation.
  //
  const double inverseSpacing = 1. / _spacing;
  for (std::size_t i = 0; i != D; ++i) {
    // Convert the coordinate to a key. We check that the translation is
    // inside the allowed domain.
    // Check the lower bound.
    assert(translation[i] >= _lower[i]);
    const Key n = Key((translation[i] - _lower[i]) * inverseSpacing);
    // Check the upper bound.
    assert(n < std::size_t(1 << _bitsPerTranslation[i]));
    key <<= _bitsPerTranslation[i];
    key |= n;
  }

  return key;
}


template<std::size_t _SubdivisionLevels, typename _Key>
inline
void
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
decode(Key key, Point* x, Point* y, Point* translation) const
{
  // We unpack in the reverse order that we packed.

  //
  // Unpack the translation.
  //
  for (std::size_t i = D; i-- != 0;) {
    const Key translationMask = ((1 << _bitsPerTranslation[i]) - 1);
    // The center of the cell.
    (*translation)[i] = _lower[i] + ((key & translationMask) + 0.5) *
                        _spacing;
    key >>= _bitsPerTranslation[i];
  }

  //
  // Unpack the rotation.
  //
  *y = decode(key);
  key >>= BitsPerAxis;
  *x = decode(key);
  key >>= BitsPerAxis;
  assert(key == 0);
  // Make y orthogonal to x.
  *y -= ext::dot(*x, *y)** x;
#ifdef STLIB_DEBUG
  assert(numerical::areEqual(ext::dot(*x, *y), 0., 10.));
#endif
  // Re-normalize the rotated y axis.
  ext::normalize(y);
}


//
// Encode/Decode for a single rotated axis.
//


template<std::size_t _SubdivisionLevels, typename _Key>
inline
typename SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::Key
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
encode(const Point& axis) const
{
#ifdef STLIB_DEBUG
  // The axis vector must be normal.
  assert(numerical::areEqual(ext::squaredMagnitude(axis), 1., 100.));
#endif
  Key key = 0;

  // Encode the octant. The z axis is the most significant bit.
  for (std::size_t i = D; i-- != 0;) {
    key <<= 1;
    if (axis[i] >= 0) {
      key |= Key(1);
    }
  }
  // Subdivide to encode the spherical triangle.
  for (std::size_t i = 0; i != _SubdivisionLevels; ++i) {
    // Pick the closest center.
    double minDistance = std::numeric_limits<double>::infinity();
    Key closest = 0;
    for (std::size_t j = 0; j != 4; ++j) {
      double d = ext::squaredDistance(axis, _centroids[i][4 * key + j]);
      if (d < minDistance) {
        minDistance = d;
        closest = j;
      }
    }
    // Encode the index in the subdivided spherical triangle.
    key <<= 2;
    key |= closest;
  }
  return key;
}


template<std::size_t _SubdivisionLevels, typename _Key>
inline
typename SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::Point
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
decode(Key key) const
{
  // Extract the portion of the key that encodes a single axis.
  const Key mask = (Key(1) << BitsPerAxis) - 1;
  key &= mask;
  // Return the appropriate centroid.
  return _centroids[_SubdivisionLevels - 1][key];
}


// Report the list of all of the keys in a neighborhood of the specified key.
template<std::size_t _SubdivisionLevels, typename _Key>
inline
void
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
neighbors(Key key, std::vector<Key>* result) const
{
  // Unpack the transformation.
  // First the translation as the indices in the grid.
  std::array<std::size_t, D> translation;
  for (std::size_t i = D; i-- != 0;) {
    const Key translationMask = ((1 << _bitsPerTranslation[i]) - 1);
    translation[i] = key & translationMask;
    key >>= _bitsPerTranslation[i];
  }
  // Then unpack the triangle indices for the rotated x and y axes.
  const Key axisMask = (1 << BitsPerAxis) - 1;
  const Key y = key & axisMask;
  key >>= BitsPerAxis;
  const Key x = key & axisMask;
  key >>= BitsPerAxis;
  assert(key == 0);

  // Get the neighbors of the translation.
  std::vector<Key> translationNeighbors;
  reportNeighbors(translation, &translationNeighbors);
  // Get the neighbors of the rotated x and y axes.
  std::vector<Key> xNeighbors, yNeighbors;
  reportNeighbors(x, &xNeighbors);
  reportNeighbors(y, &yNeighbors);

  const std::size_t BitsPerTranslation = ext::sum(_bitsPerTranslation);
  result->clear();
  result->resize(xNeighbors.size() * yNeighbors.size() *
                 translationNeighbors.size());
  std::size_t n = 0;
  // Loop over the neighbors and pack them into keys.
  for (std::size_t i = 0; i != xNeighbors.size(); ++i) {
    // First the rotated x axis.
    Key a = xNeighbors[i];
    for (std::size_t j = 0; j != yNeighbors.size(); ++j) {
      // Secondly the rotated y axis.
      Key b = a;
      b <<= BitsPerAxis;
      b |= yNeighbors[j];
      for (std::size_t k = 0; k != translationNeighbors.size(); ++k) {
        // Finally the translation.
        Key c = b;
        c <<= BitsPerTranslation;
        c |= translationNeighbors[k];
        // Record the neighboring transformation.
        (*result)[n++] = c;
      }
    }
  }
}


// Report the translations that neighbor specified one.
template<std::size_t _SubdivisionLevels, typename _Key>
inline
void
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
reportNeighbors(const std::array<std::size_t, D>& cell,
                std::vector<Key>* neighbors) const
{
  // Compute the closed lower and open upper bound.
  std::array<std::size_t, D> lower, upper;
  for (std::size_t i = 0; i != D; ++i) {
    if (cell[i] != 0) {
      lower[i] = cell[i] - 1;
    }
    else {
      lower[i] = 0;
    }
    const std::size_t size = 1 << _bitsPerTranslation[i];
#ifdef STLIB_DEBUG
    assert(cell[i] < size);
#endif
    upper[i] = std::min(cell[i] + 2, size);
  }

  // Loop over the neighboring cells.
  neighbors->clear();
  std::array<std::size_t, D> index;
  for (index[0] = lower[0]; index[0] != upper[0]; ++index[0]) {
    Key a = index[0];
    for (index[1] = lower[1]; index[1] != upper[1]; ++index[1]) {
      Key b = a;
      b <<= _bitsPerTranslation[0];
      b |= index[1];
      for (index[2] = lower[2]; index[2] != upper[2]; ++index[2]) {
        Key c = b;
        c <<= _bitsPerTranslation[1];
        c |= index[2];
        neighbors->push_back(c);
      }
    }
  }
}


// Report the triangles that neighbor the specified triangle.
template<std::size_t _SubdivisionLevels, typename _Key>
inline
void
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
reportNeighbors(const std::size_t triangle, std::vector<Key>* neighbors)
const
{
  std::set<Key> incident;
  // For each the three vertices.
  for (std::size_t i = 0; i != 3; ++i) {
    reportIncident(triangle, i, &incident);
  }
  neighbors->clear();
  for (std::set<std::size_t>::const_iterator i = incident.begin();
       i != incident.end(); ++i) {
    neighbors->push_back(*i);
  }
}


// Note that this is not a general purpose routine. It only works for
// the specified subdivision mesh used here. We use the property that
// ever vertex has an even number of incident triangles and that the
// local numbering of vertices is consistent.
template<std::size_t _SubdivisionLevels, typename _Key>
inline
void
SpecialEuclideanCode<3, _SubdivisionLevels, _Key>::
reportIncident(const std::size_t triangle, const std::size_t face,
               std::set<Key>* incident) const
{
  const std::size_t f1 = (face + 1) % 3;
  const std::size_t f2 = (face + 2) % 3;
  Key t = triangle;

  do {
    incident->insert(t);
    t = _adjacencies[t][f1];
    incident->insert(t);
    t = _adjacencies[t][f2];
  }
  while (t != triangle);
}


} // namespace geom
}
