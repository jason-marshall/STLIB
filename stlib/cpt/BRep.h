// -*- C++ -*-

#if !defined(__cpt_BRep_h__)
#define __cpt_BRep_h__

// Local
#include "stlib/cpt/Grid.h"
#include "stlib/cpt/Vertex.h"
#include "stlib/cpt/Edge.h"
#include "stlib/cpt/Face.h"

// Performance
#include "stlib/performance/Performance.h"

// Geometry
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/geometry.h"
#include "stlib/geom/mesh/iss/quality.h"
#include "stlib/geom/mesh/iss/set.h"
#include "stlib/geom/polytope/IndexedEdgePolyhedron.h"

// Std extensions.
#include "stlib/ext/array.h"

#include <fstream>
#include <vector>
#include <utility>
#include <set>

namespace stlib
{
namespace cpt
{

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

/*!
  \file BRep.h
  \brief Implements a class for a b-rep.
*/

template < std::size_t N, typename T = double >
class BRep;


//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
// 1-D
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------

//! A class for a b-rep.
template <typename T>
class BRep<1, T>
{
public:

  //
  // Public types.
  //

  //! The number type.
  typedef T Number;
  //! A point in 1-D.
  typedef std::array<Number, 1> Point;
  //! A bounding box.
  typedef geom::BBox<Number, 1> BBox;
  //! The grid type.
  typedef cpt::Grid<1, Number> Grid;

  // CONTINUE: With the Intel compiler, private members are not accessible
  // in nested classes.
#ifdef __INTEL_COMPILER
public:
#else
private:
#endif

  // The representation of a face.
  class FaceRep
  {
  private:

    //! The location of the face.
    Point _location;
    //! The orientation of the face.
    int _orientation;
    //! The identifier of the face.
    std::size_t _identifier;

  public:

    //------------------------------------------------------------------------
    //! \name Constructors etc.
    //@{

    //! Default constructor.  Uninitialized values.
    FaceRep() {}

    //! Make a 1-D face.
    FaceRep(const Point& location, const int orientation,
            const std::size_t identifier) :
      _location(location),
      _orientation(orientation),
      _identifier(identifier) {}

    //! Copy constructor.
    FaceRep(const FaceRep& other) :
      _location(other._location),
      _orientation(other._orientation),
      _identifier(other._identifier) {}

    //! Assignment operator.
    FaceRep&
    operator=(const FaceRep& other)
    {
      if (&other != this) {
        _location = other._location;
        _orientation = other._orientation;
        _identifier = other._identifier;
      }
      return *this;
    }

    //@}
    //------------------------------------------------------------------------
    //! \name Accessors.
    //@{

    //! Return the location of the face.
    const Point&
    getLocation() const
    {
      return _location;
    }

    //! Return the orientation of the face.
    int
    getOrientation() const
    {
      return _orientation;
    }

    //! Return the identifier of the face.
    std::size_t
    getIdentifier() const
    {
      return _identifier;
    }

    //@}
  };

  // Functor for comparing faces by their location.
  class LocationCompare :
    public std::binary_function<FaceRep, FaceRep, bool>
  {
  private:
    typedef std::binary_function<FaceRep, FaceRep, bool> Base;
  public:
    typedef typename Base::first_argument_type first_argument_type;
    typedef typename Base::second_argument_type second_argument_type;
    typedef typename Base::result_type result_type;

    result_type
    operator()(const first_argument_type& x, const second_argument_type& y)
    {
      return x.getLocation() < y.getLocation();
    }
  };

private:

  // CONTINUE REMOVE
  // This class needs access to private type inside BRep.
  //friend class LocationCompare;

  // Functor for comparing faces by their identifier.
  class IdentifierCompare :
    public std::binary_function<FaceRep, FaceRep, bool>
  {
  private:
    typedef std::binary_function<FaceRep, FaceRep, bool> Base;
  public:
    typedef typename Base::first_argument_type first_argument_type;
    typedef typename Base::second_argument_type second_argument_type;
    typedef typename Base::result_type result_type;

    result_type
    operator()(const first_argument_type& x, const second_argument_type& y)
    {
      return x.getIdentifier() < y.getIdentifier();
    }
  };

  // CONTINUE REMOVE
  // This class needs access to private type inside BRep.
  //friend class IdentifierCompare;

  //
  // Private types.
  //

  //! A face that is useful for computing distance and scan conversion.
  typedef cpt::Face<1, Number> Face;

  //
  // Member data.
  //

  // The faces.
  std::vector<FaceRep> _faces;
  // How far to compute the distance.
  mutable Number _maximumDistance;

public:

  //--------------------------------------------------------------------------
  // \name Constructors, etc.
  //@{

  //! Default constructor.  An empty b-rep.
  BRep() :
    _faces(),
    _maximumDistance(0) {}

  //! Construct from the faces of the b-rep.  Throw away irrelevant ones.
  /*!
    \param locations is the vector of the face locations.
    \param orientations is the vector of face orientations.
    +1 means that positive distances are to the right.  -1 means that
    positive distances are to the left.
    \param cartesianDomain is the domain of the grid.
    \param maximumDistance is how far the distance is being computed.

    Clip the b-rep so that faces outside the relevant Cartesian domain
    are thrown away.

    This constructor calls make() with the same arguments.
  */
  BRep(const std::vector<Number>& locations,
       const std::vector<int>& orientations,
       const BBox& cartesianDomain,
       const Number maximumDistance)
  {
    make(locations, orientations, cartesianDomain, maximumDistance);
  }

  //! Copy constructor.
  BRep(const BRep& other) :
    _faces(other._faces),
    _maximumDistance(other._maximumDistance) {}

  //! Assignment operator.
  BRep&
  operator=(const BRep& other)
  {
    if (&other != this) {
      _faces = other._faces;
      _maximumDistance = other._maximumDistance;
    }
    return *this;
  }

  //! Make from the faces of the b-rep.
  /*!
    \param locations is the vector of the face locations.
    \param orientations is the vector of face orientations.
    +1 means that positive distances are to the right.  -1 means that
  */
  template <typename NumberInputIter, typename IntegerInputIter>
  void
  make(const std::vector<Number>& locations,
       const std::vector<int>& orientations)
  {
#ifdef STLIB_DEBUG
    assert(locations.size() == orientations.size());
#endif
    // Add the faces.
    for (std::size_t i = 0; i != locations.size(); ++i) {
      insertFace(locations[i], orientations[i], i);
    }
    // Sort the faces.
    LocationCompare comp;
    std::sort(_faces.begin(), _faces.end(), comp);
  }

  //! Make from the faces of the b-rep.  Throw away irrelevant ones.
  /*!
    \param locations is the vector of the face locations.
    \param orientations is the vector of face orientations.
    +1 means that positive distances are to the right.  -1 means that
    \param cartesianDomain is the domain of the grid.
    \param maximumDistance is how far the distance is being computed.

    Clip the b-rep so that faces outside the relevant Cartesian domain
    are thrown away.
  */
  template <typename NumberInputIter, typename IntegerInputIter>
  void
  make(const std::vector<Number>& locations,
       const std::vector<int>& orientations,
       const BBox& cartesianDomain,
       const Number maximumDistance)
  {
#ifdef STLIB_DEBUG
    assert(locations.size() == orientations.size());
#endif

    _maximumDistance = maximumDistance;
    const BBox
    interestDomain(cartesianDomain.getLowerCorner() - maximumDistance,
                   cartesianDomain.getUpperCorner() + maximumDistance);

    // Add the faces.
    std::array<Number, 1> loc;
    for (std::size_t i = 0; i != locations.size(); ++i) {
      loc[0] = locations[i];
      if (interestDomain.isIn(loc)) {
        insertFace(locations[i], orientations[i], i);
      }
    }

    // Sort the faces.
    LocationCompare comp;
    std::sort(_faces.begin(), _faces.end(), comp);
  }

  //@}
  //--------------------------------------------------------------------------
  // \name Size accessors.
  //@{

  //! Return true if there are no faces.
  bool
  isEmpty() const
  {
    return _faces.empty();
  }

  //! Return the number of faces.
  std::size_t
  getSimplicesSize() const
  {
    return _faces.size();
  }

  //@}
  //--------------------------------------------------------------------------
  // \name Mathematical Operations.
  //@{

  //! Calculate the closest point transform to this BRep.
  std::pair<std::size_t, std::size_t>
  computeClosestPoint(std::vector<Grid>& grids,
                      const Number maximumDistance) const
  {
    // CONTINUE: Make efficient.
    assert(grids->size() > 0);

    _maximumDistance = maximumDistance;
    std::size_t i;
    std::size_t scanConversionCount = 0;
    std::size_t distanceCount = 0;
    std::pair<std::size_t, std::size_t> faceCount;

    // Find the closest points and distance for the faces.
    Face face;
    for (i = 0; i != getSimplicesSize(); ++i) {
      // Get the i_th face.
      getFace(i, &face);
      // For each grid.
      for (std::size_t n = 0; n != grids->size(); ++n) {
        // Scan convert the grid points and compute the distance etc.
        faceCount = grids[n].computeClosestPointTransform(face,
                    maximumDistance);
        scanConversionCount += faceCount.first;
        distanceCount += faceCount.second;
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Calculate the closest point transform with unsigned distance to this BRep.
  std::pair<std::size_t, std::size_t>
  computeClosestPointUnsigned(std::vector<Grid>& grids,
                              const Number maximumDistance) const
  {
    // CONTINUE: Make efficient.
    assert(grids->size() > 0);

    _maximumDistance = maximumDistance;
    std::size_t i;
    std::size_t scanConversionCount = 0;
    std::size_t distanceCount = 0;
    std::pair<std::size_t, std::size_t> faceCount;

    // Find the closest points and distance for the faces.
    Face face;
    for (i = 0; i != getSimplicesSize(); ++i) {
      // Get the i_th face.
      getFace(i, &face);
      // For each grid.
      for (std::size_t n = 0; n != grids->size(); ++n) {
        // Scan convert the grid points and compute the distance etc.
        faceCount = grids[n].computeClosestPointTransformUnsigned
                    (face, maximumDistance);
        scanConversionCount += faceCount.first;
        distanceCount += faceCount.second;
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Return the bounding box that contains the mesh.
  BBox
  computeBBox() const
  {
    if (getSimplicesSize() == 0) {
      return BBox(0, -1);
    }
    BBox box(_faces[0].getLocation()[0], _faces[0].getLocation()[0]);
    for (std::size_t n = 1; n != getSimplicesSize(); ++n) {
      box.add(_faces[n].getLocation());
    }
    return box;
  }

  //@}
  //--------------------------------------------------------------------------
  // \name File I/O.
  //@{

  //! Display information about the b-rep.
  /*!
    Report if the manifold is closed.
  */
  void
  displayInformation(std::ostream& out) const
  {
    out << "Number of faces: " << getSimplicesSize() << '\n'
        << "Bounding box: " << computeBBox() << '\n';
  }

  //! Display the b-rep.
  void
  display(std::ostream& out) const
  {
    out << "Number of faces: " << getSimplicesSize() << '\n';
    for (std::size_t i = 0; i != _faces.size(); ++i) {
      out << _faces[i].getLocation() << " "
          << _faces[i].getOrientation() << " "
          << _faces[i].getIdentifier() << '\n';
    }
  }

  //@}

private:

  //
  // Accessors.
  //

  // Make the n_th face
  void
  getFace(const std::size_t n, Face* face) const
  {
#ifdef STLIB_DEBUG
    assert(0 <= n && n < getSimplicesSize());
#endif
    // Get locations for the left and right neighbors.
    // If there is no neighbor, give a far-away point.
    // I divide by 2 to avoid overflow problems in Face.
    Point left = - std::numeric_limits<Number>::max() / 2.0;
    if (n != 0) {
      left = _faces[n - 1].getLocation();
    }
    Point right = std::numeric_limits<Number>::max() / 2.0;
    if (n != getSimplicesSize() - 1) {
      right = _faces[n + 1].getLocation();
    }
    // Make the face.
    face->make(_faces[n].getLocation(), _faces[n].getOrientation(),
               _faces[n].getIdentifier(), left, right, _maximumDistance);
  }

  std::size_t
  getFaceIdentifier(const std::size_t n) const
  {
    return _faces[n].getIdentifier();
  }

  //
  // Manipulators.
  //

  // Add a face.
  // a, b and c are indices of vertices in the positive orientation.
  void
  insertFace(const Point& location, const int orientation,
             const std::size_t faceIdentifier)
  {
    _faces.push_back(FaceRep(location, orientation, faceIdentifier));
  }

  // Clear all the member data.
  void
  clear()
  {
    _faces.clear();
  }

};


//
// File IO
//

//! Write the b-rep.
/*!
  \relates BRep<N,T>
  CONTINUE
*/
template<std::size_t N, typename T>
inline
std::ostream&
operator<<(std::ostream& out, const BRep<N, T>& br)
{
  br.display(out);
  return out;
}

} // namespace cpt
}

#define __cpt_BRep3_ipp__
#include "stlib/cpt/BRep3.ipp"
#undef __cpt_BRep3_ipp__

#define __cpt_BRep2_ipp__
#include "stlib/cpt/BRep2.ipp"
#undef __cpt_BRep2_ipp__

#endif
