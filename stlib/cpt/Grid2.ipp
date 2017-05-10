// -*- C++ -*-

#if !defined(__Grid2_ipp__)
#error This file is an implementation detail of the class Grid
#endif

namespace stlib
{
namespace cpt
{

//! A class to hold the 2-D grid data.
template<typename T>
class Grid<2, T> :
  public GridBase<2, T>
{
  //
  // Private types.
  //

private:

  typedef GridBase<2, T> Base;

  //
  // Public types.
  //

public:

  //! The number type.
  typedef typename Base::Number Number;
  //! A point in 2-D.
  typedef typename Base::Point Point;
  //! An extent in 2-D.
  typedef typename Base::SizeList SizeList;
  //! A multi-index in 2-D.
  typedef typename Base::IndexList IndexList;
  //! A single index.
  typedef typename Base::Index Index;
  //! A multi-index range in 2-D.
  typedef typename Base::Range Range;
  //! An index range iterator.
  typedef typename Base::MultiIndexRangeIterator MultiIndexRangeIterator;
  //! A lattice.
  typedef typename Base::Lattice Lattice;

  // CONTINUE: remove these when the closest point functions are templated.
  //! A vertex in the b-rep.
  typedef cpt::Vertex<2, Number> Vertex;
  //! A face in the b-rep.
  typedef cpt::Face<2, Number> Face;

public:

  //
  // Using.
  //

  // Accessors.

  //! Return the grid extents.
  using Base::getExtents;
  //! Return the grid index ranges.
  using Base::getRanges;
  //! Return true if the grids are empty.
  using Base::isEmpty;
  //! Return a const reference to the distance grid.
  using Base::getDistance;
  //! Return a const reference to the gradient of the distance grid.
  using Base::getGradientOfDistance;
  //! Return a const reference to the closest point grid.
  using Base::getClosestPoint;
  //! Return a const reference to the closest face grid.
  using Base::getClosestFace;
  //! Is the gradient of the distance being computed?
  using Base::isGradientOfDistanceBeingComputed;
  //! Is the closest point being computed?
  using Base::isClosestPointBeingComputed;
  //! Is the closest face being computed?
  using Base::isClosestFaceBeingComputed;

  // Manipulators names are already brought in with the accessors.

  // Return a reference to the distance grid.
  //using Base::getDistance;
  // Return a reference to the gradient of the distance grid.
  //using Base::getGradientOfDistance;
  // Return a reference to the closest point grid.
  //using Base::getClosestPoint;
  // Return a reference to the closest face grid.
  //using Base::getClosestFace;

  // Mathematical operations.

  //! Initialize the grids.
  using Base::initialize;
  //! Flood fill the unsigned distance.
  using Base::floodFillUnsigned;

  // File I/O.

  using Base::put;
  using Base::displayInformation;
  using Base::countKnownDistances;
  using Base::computeMinimumAndMaximumDistances;

  // Mathematical operations.

  using Base::computeClosestPointTransform;
  using Base::computeClosestPointTransformUnsigned;

  //-------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.  Uninitialized memory.
  Grid() :
    Base() {}

  //! Copy constructor.
  Grid(const Grid& other) :
    Base(other) {}

  //! Construct from grid information.
  Grid(const SizeList& extents, const IndexList& bases,
       const bool useGradientOfDistance,
       const bool useClosestPoint,
       const bool useClosestFace) :
    Base(extents, bases, useGradientOfDistance, useClosestPoint, useClosestFace) {}

  //! Construct from the grids.
  Grid(const SizeList& extents, const IndexList& bases,
       Number* distance, Number* gradientOfDistance,
       Number* closestPoint, int* closestFace) :
    Base(extents, bases, distance, gradientOfDistance, closestPoint,
         closestFace) {}

  //! Destructor.  Does not free grid memory.
  ~Grid() {}

  //! Assignment operator.
  Grid&
  operator=(const Grid& other)
  {
    Base::operator=(other);
    return *this;
  }

  using Base::rebuild;

  //@}
  //-------------------------------------------------------------------------
  //! \name Mathematical operations.
  //@{

  //! Return true if the signed distance, closest point, etc. are valid.
  bool
  isValid(const Lattice& lattice,
          Number maximumDistance, std::size_t faceIdentifierUpperBound,
          std::size_t maximumReportedErrors = 100) const;

  //! Return true if the unsigned distance, closest point, etc. are valid.
  bool
  isValidUnsigned(const Lattice& lattice,
                  Number maximumDistance,
                  std::size_t faceIdentifierUpperBound,
                  std::size_t maximumReportedErrors = 100) const;

  //! Flood fill the signed distance.
  /*!
    If there are any points with known distance then return true and set
    the unknown distances to +- farAway.  Otherwise set all the distances
    to + farAway and return false.
  */
  bool
  floodFill(Number farAway);

  //@}
};




//
// Mathematical operations
//



template<typename T>
inline
bool
Grid<2, T>::
isValid(const Lattice& lattice, const Number maximumDistance,
        const std::size_t faceIdentifierUpperBound,
        const std::size_t maximumReportedErrors) const
{
  bool result = true;
  std::size_t numberOfErrors = 0;

  const Point HugePoint =
    ext::filled_array<Point>(std::numeric_limits<Number>::max());
  const Number Eps10 = 10 * std::numeric_limits<Number>::epsilon();

  //
  // Check the distance grid.
  //
  {
    Number d;
    const MultiIndexRangeIterator end = MultiIndexRangeIterator::end(getRanges());
    for (MultiIndexRangeIterator i = MultiIndexRangeIterator::begin(getRanges());
         i != end; ++i) {
      d =  getDistance()(*i);
      if (!(d == std::numeric_limits<Number>::max() ||
            std::abs(d) <= maximumDistance)) {
        std::cerr << "In Grid<2,T>::isValid():" << '\n'
                  << "    Bad distance value." << '\n'
                  << "    d = " << d << '\n'
                  << "    (i,j) = " << *i << '\n';
        result = false;
        if (++numberOfErrors >= maximumReportedErrors) {
          std::cerr << "Maximum number of errors exceeded." << '\n';
          return false;
        }
      }
    }
  }

  // Check the numerical derivative of distance in the x direction.
  {
    Number d1, d2;
    const Number deltaX = lattice.getDelta()[0] *
                          (1 + std::sqrt(std::numeric_limits<Number>::epsilon()));

    SizeList extents = getRanges().extents();
    extents[0] -= std::size_t(1);
    const IndexList bases = getRanges().bases();
    const MultiIndexRangeIterator end =
      MultiIndexRangeIterator::end(Range(extents, bases));
    IndexList j;
    for (MultiIndexRangeIterator i =
           MultiIndexRangeIterator::begin(Range(extents, bases)); i != end; ++i) {
      // j = i + (1,0)
      j = *i;
      j[0] += Index(1);
      d1 =  getDistance()(*i);
      d2 =  getDistance()(j);
      if (d1 != std::numeric_limits<Number>::max() &&
          d2 != std::numeric_limits<Number>::max()) {
        if (std::abs(d1 - d2) > deltaX) {
          std::cerr << "In Grid<2,T>::isValid():\n"
                    << "    Bad distance difference in x direction.\n"
                    << "    d1 = " << d1 << "  d2 = " << d2 << '\n'
                    << "    std::abs(d1 - d2) = " << std::abs(d1 - d2) << '\n'
                    << "    (i,j) = " << *i << '\n'
                    << "    deltaX = " << deltaX << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded." << '\n';
            return false;
          }
        }
      }
    }
  }

  // Check the numerical derivative of distance in the y direction.
  {
    Number d1, d2;
    const Number deltaY = lattice.getDelta()[1] *
                          (1 + std::sqrt(std::numeric_limits<Number>::epsilon()));

    SizeList extents = getRanges().extents();
    extents[1] -= std::size_t(1);
    const IndexList bases = getRanges().bases();
    const MultiIndexRangeIterator end =
      MultiIndexRangeIterator::end(Range(extents, bases));
    IndexList j;
    for (MultiIndexRangeIterator i =
           MultiIndexRangeIterator::begin(Range(extents, bases)); i != end; ++i) {
      // j = i + (0,1)
      j = *i;
      j[1] += Index(1);
      d1 =  getDistance()(*i);
      d2 =  getDistance()(j);
      if (d1 != std::numeric_limits<Number>::max() &&
          d2 != std::numeric_limits<Number>::max()) {
        if (std::abs(d1 - d2) > deltaY) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Bad distance difference in y direction.\n"
                    << "    d1 = " << d1 << "  d2 = " << d2 << '\n'
                    << "    std::abs(d1 - d2) = " << std::abs(d1 - d2) << '\n'
                    << "    (i,j) = " << *i << '\n'
                    << "    deltaY = " << deltaY << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded." << '\n';
            return false;
          }
        }
      }
    }
  }

  //
  // Check the closest face grid.
  //

  // If the closest face is being computed.
  if (isClosestFaceBeingComputed()) {
    Number d;
    Point cp;

    std::size_t face;
    const MultiIndexRangeIterator end = MultiIndexRangeIterator::end(getRanges());
    for (MultiIndexRangeIterator i = MultiIndexRangeIterator::begin(getRanges());
         i != end; ++i) {
      face = getClosestFace()(*i);
      if (face >= faceIdentifierUpperBound &&
          face != std::numeric_limits<std::size_t>::max()) {
        std::cerr << "In Grid<2,T>::isValid():\n"
                  << "    Bad closest face value.\n"
                  << "    (i,j) = " << *i << '\n'
                  << "    face = " << face << '\n'
                  << "    faceIdentifierUpperBound = "
                  << faceIdentifierUpperBound << '\n';
        result = false;
        if (++numberOfErrors >= maximumReportedErrors) {
          std::cerr << "Maximum number of errors exceeded.\n";
          return false;
        }
      }

      d =  getDistance()(*i);
      if (face == std::numeric_limits<std::size_t>::max()) {
        if (d != std::numeric_limits<Number>::max()) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Face is unknown but distance is not huge.\n"
                    << "    (i,j) = " << *i << '\n'
                    << "    distance = " << d << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }

        if (isClosestPointBeingComputed()) {
          cp = getClosestPoint()(*i);
          if (cp != HugePoint) {
            std::cerr
                << "In Grid::isValid():\n"
                << "    Face is unknown but closest point is not huge.\n"
                << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // if (face == std::numeric_limits<std::size_t>::max())
      else { // 0 <= face < faceIdentifierUpperBound
        if (std::abs(d) > maximumDistance) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Face is known, distance is too big.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }

        if (isClosestPointBeingComputed()) {
          cp = getClosestPoint()(*i);
          if (cp == HugePoint) {
            std::cerr << "In Grid::isValid():\n"
                      << "    Face is known but closest point is huge.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // else
    } // for i
  } // if (isClosestFaceBeingComputed())

  //
  // Check the closest point grid.
  //

  // If the closest point is being computed.
  if (isClosestPointBeingComputed()) {
    Number d;
    Point cp;
    std::size_t face;

    const MultiIndexRangeIterator end = MultiIndexRangeIterator::end(getRanges());
    for (MultiIndexRangeIterator i = MultiIndexRangeIterator::begin(getRanges());
         i != end; ++i) {
      cp = getClosestPoint()(*i);
      d =  getDistance()(*i);
      if (cp == HugePoint) {
        if (d != std::numeric_limits<Number>::max()) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Closest pt is huge, distance is not huge.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }

        if (isClosestFaceBeingComputed()) {
          face = getClosestFace()(*i);
          if (face != std::numeric_limits<std::size_t>::max()) {
            std::cerr << "In Grid::isValid():\n"
                      << "    Closest pt is huge but face is known.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // if (cp == HugePoint)
      else {
        if (std::abs(d) > maximumDistance) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Closest pt is known, distance is too big.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }

        if (isClosestFaceBeingComputed()) {
          face = getClosestFace()(*i);
          if (face == std::numeric_limits<std::size_t>::max()) {
            std::cerr << "In Grid::isValid():\n"
                      << "    Closest pt is known but face is unknown.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // else
    } // for i
  } // if (isClosestPointBeingComputed())

  //
  // Check the gradient of the distance grid.
  //

  // If the gradient of the distance is being computed.
  if (isGradientOfDistanceBeingComputed()) {
    Number d;
    Point gd, cp, position;
    std::size_t face;

    const MultiIndexRangeIterator end = MultiIndexRangeIterator::end(getRanges());
    for (MultiIndexRangeIterator i = MultiIndexRangeIterator::begin(getRanges());
         i != end; ++i) {
      gd = getGradientOfDistance()(*i);
      d =  getDistance()(*i);
      if (gd == HugePoint) {
        if (d != std::numeric_limits<Number>::max()) {
          std::cerr << "In Grid<2,T>::isValid():\n"
                    << "    Grad dist is huge but distance is not huge.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }
        if (isClosestFaceBeingComputed()) {
          face = getClosestFace()(*i);
          if (face != std::numeric_limits<std::size_t>::max()) {
            std::cerr << "In Grid<2,T>::isValid():\n"
                      << "    Grad dist is huge but face is known.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // if (gd == HugePoint)
      else {
        if (std::abs(d) > maximumDistance) {
          std::cerr << "In Grid<2,T>::isValid():\n"
                    << "    Grad dist is known, distance is too big.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded."
                      << '\n';
            return false;
          }
        }

        if (isClosestFaceBeingComputed()) {
          face = getClosestFace()(*i);
          if (face == std::numeric_limits<std::size_t>::max()) {
            std::cerr << "In Grid<2,T>::isValid():\n"
                      << "    Grad dist is known but face is unknown.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
        // Check the magnitude of the gradient of the distance.
        if (std::abs(ext::magnitude(gd) - 1) > Eps10) {
          std::cerr << "In Grid<2,T>::isValid():\n"
                    << "    Magnitude of gradient, " << ext::magnitude(gd)
                    << ", is not unity.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }
        // Check the direction of the gradient of the distance.
        if (isClosestPointBeingComputed()) {
          position[0] = (*i)[0];
          position[1] = (*i)[1];
          lattice.convertIndexToLocation(&position);
          cp = getClosestPoint()(*i);
          if (std::abs(ext::discriminant(position - cp, gd)) >
              std::sqrt(std::numeric_limits<Number>::epsilon())) {
            std::cerr << "In Grid<2,T>::isValid():\n"
                      << "    Direction of gradient is wrong.\n"
                      << "    gradient = " << gd << '\n'
                      << "    position - closest point = "
                      << position - cp << '\n'
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // else
    } // for i
  } // if (isGradientOfDistanceBeingComputed())

  return result;
}


template<typename T>
inline
bool
Grid<2, T>::
isValidUnsigned(const Lattice& lattice,
                const Number maximumDistance,
                const std::size_t faceIdentifierUpperBound,
                const std::size_t maximumReportedErrors) const
{
  bool result = true;
  std::size_t numberOfErrors = 0;

  const Point HugePoint =
    ext::filled_array<Point>(std::numeric_limits<Number>::max());
  const Number Eps10 = 10 * std::numeric_limits<Number>::epsilon();

  //
  // Check the distance grid.
  //
  {
    Number d;
    const MultiIndexRangeIterator end = MultiIndexRangeIterator::end(getRanges());
    for (MultiIndexRangeIterator i = MultiIndexRangeIterator::begin(getRanges());
         i != end; ++i) {
      d =  getDistance()(*i);
      if (!(d == std::numeric_limits<Number>::max() ||
            (0 <= d && d <= maximumDistance))) {
        std::cerr << "In Grid::isValid():\n"
                  << "    Bad distance value.\n"
                  << "    d = " << d << '\n'
                  << "    (i,j) = " << *i << '\n';
        result = false;
        if (++numberOfErrors >= maximumReportedErrors) {
          std::cerr << "Maximum number of errors exceeded.\n";
          return false;
        }
      }
    }
  }

  // Check the numerical derivative of distance in the x direction.
  {
    Number d1, d2;
    const Number deltaX = lattice.getDelta()[0] *
                          (1 + std::sqrt(std::numeric_limits<Number>::epsilon()));

    SizeList extents = getRanges().extents();
    extents[0] -= std::size_t(1);
    const IndexList bases = getRanges().bases();
    const MultiIndexRangeIterator end =
      MultiIndexRangeIterator::end(Range(extents, bases));
    IndexList j;
    for (MultiIndexRangeIterator i =
           MultiIndexRangeIterator::begin(Range(extents, bases)); i != end; ++i) {
      // j = i + (1,0)
      j = *i;
      j[0] += Index(1);
      d1 =  getDistance()(*i);
      d2 =  getDistance()(j);
      if (d1 != std::numeric_limits<Number>::max() &&
          d2 != std::numeric_limits<Number>::max()) {
        if (std::abs(d1 - d2) > deltaX) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Bad distance difference in x direction.\n"
                    << "    d1 = " << d1 << "  d2 = " << d2 << '\n'
                    << "    std::abs(d1 - d2) = " << std::abs(d1 - d2) << '\n'
                    << "    (i,j) = " << *i << '\n'
                    << "    deltaX = " << deltaX << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }
      }
    }
  }

  // Check the numerical derivative of distance in the y direction.
  {
    Number d1, d2;
    const Number deltaY = lattice.getDelta()[1] *
                          (1 + std::sqrt(std::numeric_limits<Number>::epsilon()));

    SizeList extents = getRanges().extents();
    extents[1] -= std::size_t(1);
    const IndexList bases = getRanges().bases();
    const MultiIndexRangeIterator end =
      MultiIndexRangeIterator::end(Range(extents, bases));
    IndexList j;
    for (MultiIndexRangeIterator i =
           MultiIndexRangeIterator::begin(Range(extents, bases)); i != end; ++i) {
      // j = i + (0,1)
      j = *i;
      j[1] += Index(1);
      d1 =  getDistance()(*i);
      d2 =  getDistance()(j);
      if (d1 != std::numeric_limits<Number>::max() &&
          d2 != std::numeric_limits<Number>::max()) {
        if (std::abs(d1 - d2) > deltaY) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Bad distance difference in y direction.\n"
                    << "    d1 = " << d1 << "  d2 = " << d2 << '\n'
                    << "    std::abs(d1 - d2) = " << std::abs(d1 - d2) << '\n'
                    << "    (i,j) = " << *i << '\n'
                    << "    deltaY = " << deltaY << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }
      }
    }
  }

  //
  // Check the closest face grid.
  //

  // If the closest face is being computed.
  if (isClosestFaceBeingComputed()) {
    Number d;
    Point cp;

    std::size_t face;
    const MultiIndexRangeIterator end = MultiIndexRangeIterator::end(getRanges());
    for (MultiIndexRangeIterator i = MultiIndexRangeIterator::begin(getRanges());
         i != end; ++i) {
      face = getClosestFace()(*i);
      if (face >= faceIdentifierUpperBound &&
          face != std::numeric_limits<std::size_t>::max()) {
        std::cerr << "In Grid::isValid():\n"
                  << "    Bad closest face value.\n"
                  << "    (i,j) = " << *i << '\n'
                  << "    face = " << face << '\n'
                  << "    faceIdentifierUpperBound = "
                  << faceIdentifierUpperBound << '\n';
        result = false;
        if (++numberOfErrors >= maximumReportedErrors) {
          std::cerr << "Maximum number of errors exceeded.\n";
          return false;
        }
      }

      d =  getDistance()(*i);
      if (face == std::numeric_limits<std::size_t>::max()) {
        if (d != std::numeric_limits<Number>::max()) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Face is unknown but distance is not huge.\n"
                    << "    (i,j) = " << *i << '\n'
                    << "    distance = " << d << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }

        if (isClosestPointBeingComputed()) {
          cp = getClosestPoint()(*i);
          if (cp != HugePoint) {
            std::cerr << "In Grid::isValid():\n"
                      << "    Face is unknown but closest point is not huge.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // if (face == std::numeric_limits<std::size_t>::max())
      else { // 0 <= face < faceIdentifierUpperBound
        if (d < 0 || d > maximumDistance) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Face is known, distance is out of range.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }

        if (isClosestPointBeingComputed()) {
          cp = getClosestPoint()(*i);
          if (cp == HugePoint) {
            std::cerr << "In Grid::isValid():\n"
                      << "    Face is known but closest point is huge.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // else
    } // for i
  } // if (isClosestFaceBeingComputed())

  //
  // Check the closest point grid.
  //

  // If the closest point is being computed.
  if (isClosestPointBeingComputed()) {
    Number d;
    Point cp;
    std::size_t face;

    const MultiIndexRangeIterator end = MultiIndexRangeIterator::end(getRanges());
    for (MultiIndexRangeIterator i = MultiIndexRangeIterator::begin(getRanges());
         i != end; ++i) {
      cp = getClosestPoint()(*i);
      d =  getDistance()(*i);
      if (cp == HugePoint) {
        if (d != std::numeric_limits<Number>::max()) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Closest pt is huge, distance is not huge.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }

        if (isClosestFaceBeingComputed()) {
          face = getClosestFace()(*i);
          if (face != std::numeric_limits<std::size_t>::max()) {
            std::cerr << "In Grid::isValid():\n"
                      << "    Closest pt is huge but face is known.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // if (cp == HugePoint)
      else {
        if (d < 0 || d > maximumDistance) {
          std::cerr << "In Grid::isValid():\n"
                    << "    Closest pt is known, distance is out of range.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }

        if (isClosestFaceBeingComputed()) {
          face = getClosestFace()(*i);
          if (face == std::numeric_limits<std::size_t>::max()) {
            std::cerr << "In Grid::isValid():\n"
                      << "    Closest pt is known but face is unknown.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // else
    } // for i
  } // if (isClosestPointBeingComputed())

  //
  // Check the gradient of the distance grid.
  //

  // If the gradient of the distance is being computed.
  if (isGradientOfDistanceBeingComputed()) {
    Number d;
    Point gd, cp, position;
    std::size_t face;

    const MultiIndexRangeIterator end = MultiIndexRangeIterator::end(getRanges());
    for (MultiIndexRangeIterator i = MultiIndexRangeIterator::begin(getRanges());
         i != end; ++i) {
      gd = getGradientOfDistance()(*i);
      d =  getDistance()(*i);
      if (gd == HugePoint) {
        if (d != std::numeric_limits<Number>::max()) {
          std::cerr << "In Grid<2,T>::isValid():\n"
                    << "    Grad dist is huge but distance is not huge.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }

        if (isClosestFaceBeingComputed()) {
          face = getClosestFace()(*i);
          if (face != std::numeric_limits<std::size_t>::max()) {
            std::cerr << "In Grid<2,T>::isValid():\n"
                      << "    Grad dist is huge but face is known.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // if (gd == HugePoint)
      else {
        if (d < 0 || d > maximumDistance) {
          std::cerr << "In Grid<2,T>::isValid():\n"
                    << "    Grad dist is known, distance is out of range.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }

        if (isClosestFaceBeingComputed()) {
          face = getClosestFace()(*i);
          if (face == std::numeric_limits<std::size_t>::max()) {
            std::cerr << "In Grid<2,T>::isValid():\n"
                      << "    Grad dist is known but face is unknown.\n"
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
        // Check the magnitude of the gradient of the distance.
        if (std::abs(ext::magnitude(gd) - 1) > Eps10) {
          std::cerr << "In Grid<2,T>::isValid():\n"
                    << "    Magnitude of gradient, " << ext::magnitude(gd)
                    << ", is not unity.\n"
                    << "    (i,j) = " << *i << '\n';
          result = false;
          if (++numberOfErrors >= maximumReportedErrors) {
            std::cerr << "Maximum number of errors exceeded.\n";
            return false;
          }
        }
        // Check the direction of the gradient of the distance.
        if (isClosestPointBeingComputed()) {
          position[0] = (*i)[0];
          position[1] = (*i)[1];
          lattice.convertIndexToLocation(&position);
          cp = getClosestPoint()(*i);
          if (std::abs(ext::discriminant(position - cp, gd)) >
              std::sqrt(std::numeric_limits<Number>::epsilon())) {
            std::cerr << "In Grid<2,T>::isValid():\n"
                      << "    Direction of gradient is wrong.\n"
                      << "    gradient = " << gd << '\n'
                      << "    position - closest point = "
                      << position - cp << '\n'
                      << "    (i,j) = " << *i << '\n';
            result = false;
            if (++numberOfErrors >= maximumReportedErrors) {
              std::cerr << "Maximum number of errors exceeded.\n";
              return false;
            }
          }
        }
      } // else
    } // for i
  } // if (isGradientOfDistanceBeingComputed())

  return result;
}



template<typename T>
inline
bool
Grid<2, T>::
floodFill(const Number farAway)
{
  const std::size_t N = 2;

  //
  // See if the distance is known for any grid points
  //
  bool result = false;
  int sign = 0;
  typename container::MultiArray<Number, 2>::const_iterator iter =
    getDistance().begin();
  const typename container::MultiArray<Number, 2>::const_iterator iter_end =
    getDistance().end();
  for (; !result && iter != iter_end; ++iter) {
    if (*iter != std::numeric_limits<Number>::max()) {
      result = true;
      sign = (*iter > 0) ? 1 : -1;
    }
  }

  //
  // If there are any points in a known distance.
  //
  if (result) {
    int ySign = sign;

    IndexList lower = getRanges().bases();
    IndexList upper = lower;
    for (std::size_t n = 0; n != N; ++n) {
      upper[n] += getRanges().extents()[n];
    }
    IndexList i;

    //
    // Flood fill the distance with +- farAway.
    //
    for (i[1] = lower[1]; i[1] != upper[1]; ++i[1]) {
      i[0] = lower[0];
      if (getDistance()(i) != std::numeric_limits<Number>::max()) {
        ySign = (getDistance()(i) > 0) ? 1 : -1;
      }
      sign = ySign;
      for (i[0] = lower[0]; i[0] != upper[0]; ++i[0]) {
        if (getDistance()(i) != std::numeric_limits<Number>::max()) {
          sign = (getDistance()(i) > 0) ? 1 : -1;
        }
        else {
          // Set the distance to +- farAway.
          getDistance()(i) = sign * farAway;
        }
      }
    }
  } // end if (result)
  else {
    // Set the distance to +farAway.
    getDistance().fill(farAway);
  }

  Base::copyToDistanceExternal();
  return result;
}

} // namespace cpt
}
