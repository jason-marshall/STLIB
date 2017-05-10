// -*- C++ -*-

/*!
  \file geom/spatialIndexing/SpecialEuclideanCode.h
  \brief Encode and decode transformations in the special Euclidean group.
*/

#if !defined(__geom_spatialIndexing_SpecialEuclideanCode_h__)
#define __geom_spatialIndexing_SpecialEuclideanCode_h__

#include "stlib/geom/kernel/BBox.h"

#include "stlib/numerical/equality.h"

#include <boost/math/quaternion.hpp>

#include <set>

namespace stlib
{
namespace geom
{

//! Encode and decode transformations in the special Euclidean group.
template<std::size_t _D, std::size_t _SubdivisionLevels,
         typename _Key = std::size_t>
class SpecialEuclideanCode;


//! Encode and decode transformations in the 3-D special Euclidean group.
/*!
  \par
  The <a href="http://en.wikipedia.org/wiki/Euclidean_group">Euclidean
  group</a> <em>E(n)</em> is the symmetry group of <em>n</em>-dimensional
  Euclidean space. The special Euclidean group <em>SE(n)</em> is the subset
  of these transformations that preserve orientation. This set of
  transformations is commonly called the rigid body motions.

  \par
  Any transformation in <em>SE(3)</em> can be represented with a rotation,
  which is a member of the special orthogonal group <em>SO(3)</em>, followed
  by a translation. The purpose of this class is to convert between
  transformations and spatial indices. The spacial indices are packed
  into a single integer whose type is specified as a template parameter.

  \par
  Converting a translation into a spatial index is trivial. Given a bounding
  box that covers the desired domain, one simply defines a uniform grid that
  covers the bounding box. The constructor for this class takes the bounding
  box and the grid spacing as arguments. To convert a translation,
  one calculates the indices of the grid cell that contains the point at the
  head of the translation vector. Decoding returns the center of the
  cell.

  \par
  Spatial indices for rotations are trickier due to the topology of
  <em>SO(3)</em>, which is homeomorphic to a ball of radius &pi;
  in 3-D in which
  antipodal points are identified. (For a point in the ball, the vector
  from the origin is the axis of rotation and the distance from the origin
  is the angle of rotation.) Thus we consider a representation that is more
  amenable to spatial indexing. Note that the coordinates of the <em>x</em>
  and <em>y</em> axes under the rotation uniquely identify it. (We don't
  need to specify the coordinates of the <em>z</em> axis because
  \f$z = x \times y\f$. Because the rotated axes are unit vectors, we
  introduce a spatial indexing for the unit sphere. The combined spatial
  indices for the rotated <em>x</em> and <em>y</em> axes is then the
  spatial index for the rotation.

  \par
  Now we consider the problem of generating a spatial index for unit vectors,
  i.e. we discretize the unit sphere. We tesselate the unit sphere with
  <em>n</em> spherical triangles. The index of the triangle whose centroid
  is closest to the head of the vector is its spatial index. We generate
  tesselations with different resolutions by subdividing an initial,
  coarse mesh. We start with an octahedron; each spherical triangle covers one
  octant. We use three bits to encode the octant. The initial octahedron
  depicted below. We have flattened it into a planar graph for easier labeling.
  %Triangle 0 has corners at the ends of the unit vectors in the -x, -y, and -z
  directions. The triangles are numbered according to binary number zyx where
  each digit indicates negative or positive direction in each dimension.
  Thus triangle 1 has the binary representation 001 and has corners at the
  ends of the vectors in the +x, -y, and -z directions.

  \image html OctahedronLabeled.png "The initial octahedron with labeled triangles and vertices."
  \image latex OctahedronLabeled.png "The initial octahedron with labeled triangles and vertices."

  \par
  The vertices of a triangle have local indices 0, 1, and 2.
  The local vertex numbering is consistent. That is, if a vertex has
  local index <em>n</em> in given triangle, then it has the same local
  index in the two adjacent triangles that share the vertex. Below
  we show a triangle and its three adjacent neighbors.

  \image html TriangleAdjacent.png "Adjacent triangles and local vertex numbering."
  \image latex TriangleAdjacent.png "Adjacent triangles and local vertex numbering."

  \par
  We subdivide a triangle by splitting the edges at the midpoints to
  produce four triangles.  The index of the child triangle is encoded
  with two bits. The number of subdivision levels is specified as a
  template parameter. In the figure below we subdivide a triangle
  and its adjacent neighbors (shown above). The four children of the initial
  triangle are labeled 0, 1, 2, and 3. The three children that are incident
  to vertices 0, 1, and 2 are labeled according to the incident vertex.
  The final child, which is at the center of the parent, is labeled 3.
  The vertices of the children are labeled to maintain consistency with
  adjacent neighbors. Below we show the labeling of the subdivided triangles
  and the vertices.

  \image html TriangleSubdivision.png "Labeling for subdivision."
  \image latex TriangleSubdivision.png "Labeling for subdivision."

  \par
  Now we have a method covering the unit sphere with a triangle mesh. The user
  specifies the resolution by choosing the number of subdivisions <em>s</em>
  to obtain <em>n = 8 * 4<sup>s</sup></em> triangles. The spatial index for
  a unit vector is the index of the triangle whose centroid is closest.
  The calculation of the closest centroid is straightforward. Given a
  unit vector, the sign of each coordinate determines the octant
  (which holds a single triangle in the unrefined mesh). We store the
  centroids for all triangles at each level refinement. We loop over
  the levels of refinement. At each iteration, we compute the distance
  to the centroids of four child triangles and choose the closest distance.
  Decoding a key is easy. We simply use the key as an index into the
  array of centroids at the final level of subdivision.

  \par Distance between transformations.
  Suppose that you have set of transformations. For a given transformation
  <em>t</em> you wish to determine all other transformation that are
  close. Two transformation are close if the distance between the them
  is sufficiently small. Measuring the distance between two translations is
  easy, just use the Euclidean distance between the endpoints of the vectors.
  One can also measure distance between rotations. For example, if the
  rotations are represented with unit quaternions <em>p</em> and <em>q</em>,
  then the distance between them is the minimum of the norms of
  <em>p - q</em> and <em>p + q</em>. (We need to take the minimum of these
  two quantities because the representation as a unit quaternion is not
  unique: <em>p</em> is the same rotation as <em>-p</em>.)

  \par Neighbors.
  Now consider again the problem of determining the transformations that
  are close to a specified transformation <em>t</em>. Transformations that
  share the same spatial index as <em>t</em> are close. The distance depends
  both on the spacing for the Cartesian grid and the number of subdivisions
  for the rotation mesh. Note that having the spatial index is a sufficient,
  but not a necessary condition for being close. Two transformations may be
  arbitrarily close, but have different keys. In this case the keys represent
  neighboring cells, either in the Cartesian grid, or the triangle mesh.
  Thus to to obtain all of the transformations that are close, one needs to
  examine all of the neighboring keys. In the Cartesian grid, interior cells
  have 3<sup>3</sup> = 27 cells in their neighborhood. Note that we count,
  the cell itself as being in the neighborhood. Boundary cells may have as few
  as 2<sup>3</sup> = 8 cells in their neighborhood. Next consider the
  subdivision mesh. Six of the vertices in the mesh have four incident
  triangles. All other vertices have six incident triangles. The minimum
  number of subdivision levels one. Thus a triangle has either 11 or
  13 triangles in its neighborhood. The encoding of a transformation
  is composed of the encodings of the translation and two unit vectors.
  Thus each key may have as many as 27 * 13 * 13 = 4563 keys in its
  immediate neighborhood.

  \par
  By choosing an appropriate grid spacing and an suitable number of subdivision
  levels, one can use the neighboring keys to find all of transformation that
  are close to a specified one. This approach is legitimate, but not
  the most general way of finding close transformations. For a more rigorous
  approach one would define the search distance. One would expand the search
  beyond the immediate neighbors and then compute the distance to identify
  the close transformations from the candidates.

  \par
  In the table below we collect quantitative information about the
  spatial indexing. For a given level of subdivision <em>s</em>,
  the number of bits used to encode the rotation is 2 * (3 + 2 * s).
  The number of triangles in the resulting mesh is 8 * 4<sup><em>s</em></sup>.
  The rotation is represented with the rotated x and y axes. We indicate
  the number of distinct rotations that may be represented. This is less
  that the square of the number of triangles because the rotated x and
  y axes are constrained to be orthogonal. Next we give performance
  results. The time to encode and decode a transformation are given
  in nanoseconds. Encoding gets more expensive as as the number of subdivision
  levels increases. However, the cost of decoding is roughly constant.
  This makes sense as encoding has linear computational complexity, while
  decoding has constant complexity. The cost of determining the neighbors
  of a given key is given in microseconds. Note that since this operation
  reports several thousand neighbors, the cost per neighbor is on the
  order of a few nanseconds. Thus, this operation is very efficient.

  <table>
  <tr>
  <th> Subdivision levels
  <th> Rotation Bits
  <th> Directions
  <th> Rotations
  <th> Encode (ns)
  <th> Decode (ns)
  <th> Neighbors (&mu;s)
  <tr>
  <td> 1
  <td> 10
  <td> 32
  <td> 704
  <td> 85
  <td> 73
  <td> 13
  <tr>
  <td> 2
  <td> 14
  <td> 128
  <td> 6,262
  <td> 140
  <td> 84
  <td> 16
  <tr>
  <td> 3
  <td> 18
  <td> 512
  <td> 50,418
  <td> 200
  <td> 78
  <td> 16
  </table>


  \par
  Consider the spatial index for the translation alone.
  If a translation <em>y</em> is not a neighbor of <em>x</em> then the
  minimum distance between the two is the grid spacing &delta;.
  If a translation <em>y</em> is a neighbor of <em>x</em>, then the
  maximum Euclidean distance <em>d</em> between the two is twice the diagonal
  length of a grid cell. If &delta; is the spacing, then
  \f$d \leq 2 \sqrt{3} \delta\f$. The important quantity is the ratio of
  these distance. A neighbor search returns all translations with &delta;
  of <em>x</em>, but may return translations as far away as
  \f$2 \sqrt{3} \delta\f$. One can use the neighbor search to perform
  fixed radius search, as long as the search radius is no greater than
  &delta;. Just perform the neighbor search and then compute the
  Euclidean distance to check which candidates are within desired
  radius.


  <table>
  <tr>
  <th> Subdivision levels
  <th> Farthest neighbor
  <th> Closest non-neighbor
  <tr>
  <td> 1
  <td> 1.3
  <td> 0.34
  <tr>
  <td> 2
  <td> 0.65
  <td> 0.15
  <tr>
  <td> 3
  <td> 0.34
  <td> 0.07
  <tr>
  <td> 4
  <td> 0.17
  <td> 0.023
  </table>

*/
template<std::size_t _SubdivisionLevels, typename _Key>
class SpecialEuclideanCode<3, _SubdivisionLevels, _Key>
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t D = 3;
  //! The number of bits per rotated axis.
  BOOST_STATIC_CONSTEXPR std::size_t BitsPerAxis = 3 + 2 * _SubdivisionLevels;

  //
  // Types.
  //
public:

  //! The integer type that holds the key.
  typedef _Key Key;
  //! A Cartesian coordinate.
  typedef std::array<double, D> Point;
  //! A bounding box in Cartesian space.
  typedef geom::BBox<double, D> BBox;
  //! A quaternion.
  typedef boost::math::quaternion<double> Quaternion;

protected:

  //! The list of 3 adjacent triangle indices.
  typedef std::array<std::size_t, 3> AdjacentList;

  //
  // Member data.
  //
protected:

  //! The lower corner of the Cartesian domain.
  Point _lower;
  //! The grid spacing in the Cartesion domain.
  double _spacing;
  //! For each dimension, the number of bits used to store the coordinate.
  std::array<std::size_t, D> _bitsPerTranslation;
  // The centroids of the spherical triangles at each level of subdivision.
  std::array<std::vector<Point>, _SubdivisionLevels> _centroids;
  // The adjacencies at the final level of subdivision.
  std::vector<AdjacentList> _adjacencies;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
   We use the synthesized copy construtor, assignment operator, and
   destructor. */
  //@{
public:

  //! Construct from the Cartesian domain and the grid spacing.
  SpecialEuclideanCode(const BBox& domain, double spacing);

protected:

  //! Compute the centroids of the spherical triangles at each level of subdivision.
  void
  computeCentroids();

  //! Compute the adjacencies at the finest level of subdivision.
  void
  computeAdjacencies();

  //@}
  //--------------------------------------------------------------------------
  //! \name Encode/decode.
  //@{
public:

  //! Encode the transformation.
  /*!
    \param q The unit quaternion that defines the rotation.
    \param translation The translation vector.
  */
  Key
  encode(const Quaternion& q, const Point& translation) const;

  //! Decode the transformation.
  /*!
    \param key The encoded transformation.
    \param q The unit quaternion that defines the rotation.
    \param translation The translation vector.

    \note The unit quaternions q and -q define the same rotation. While
    the real component of the calculated quaternion is set to be non-negative.
    this is not sufficient to resolve the ambiguities. Thus while the
    encoded and decoded quaternions may not be close, they represent
    approximately the same rotation.
  */
  void
  decode(Key key, Quaternion* q, Point* translation) const;

  //! Encode the transformation.
  /*!
    \param x The transformed x axis.
    \param y The transformed y axis.
    \param translation The translation vector.
  */
  Key
  encode(const Point& x, const Point& y, const Point& translation) const;

  //! Decode the transformation.
  /*!
    \param key The encoded transformation.
    \param x The transformed x axis.
    \param y The transformed y axis.
    \param translation The translation vector.
  */
  void
  decode(Key key, Point* x, Point* y, Point* translation) const;

protected:

  //! Encode the axis.
  /*!
    \param axis The transformed axis.
  */
  Key
  encode(const Point& axis) const;

  //! Decode key to get the rotated axis.
  Point
  decode(Key key) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name Encode/decode.
  //@{
public:

  //! Report the list of all of the keys in a neighborhood of the specified key.
  /*! Each of the two encoded axes have either 11 or 13 neighbors. A
    translation has between 8 and 27 neighbors. Thus the number of
    neighbors for a transformation is between 11 * 11 * 8 = 968 and
    13 * 13 * 27 = 4563.

    I use a std::vector for the output because it is significantly faster
    than using a generic output iterator. */
  void
  neighbors(Key key, std::vector<Key>* result) const;

protected:

  //! Report the translations that neighbor specified one.
  /*! This includes all grid cells that differ from the input by no more
   than one in each dimension. Thus there are 3<sup>3</sup> = 27 reported
   neighbors for translations that map to an interior cell. */
  void
  reportNeighbors(const std::array<std::size_t, D>& cell,
                  std::vector<Key>* neighbors) const;

  //! Report the triangles that neighbor the specified triangle.
  /*! This includes all of the triangles that are incident to a vertex of
    the specified triangle. Note that the triangle itself is included in
    the output. The neighbor list is first cleared, and then the neighbors
    are added. */
  void
  reportNeighbors(std::size_t triangle, std::vector<Key>* neighbors) const;

  //! Report the triangles that are incident to the specified vertex.
  void
  reportIncident(std::size_t triangle, std::size_t face,
                 std::set<Key>* incident) const;

  //@}
};


} // namespace geom
}

#define __geom_spatialIndexing_SpecialEuclideanCode_ipp__
#include "stlib/geom/spatialIndexing/SpecialEuclideanCode.ipp"
#undef __geom_spatialIndexing_SpecialEuclideanCode_ipp__

#endif
