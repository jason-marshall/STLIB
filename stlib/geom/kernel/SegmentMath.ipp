// -*- C++ -*-

#if !defined(__geom_SegmentMath_ipp__)
#error This file is an implementation detail of the class SegmentMath.
#endif

namespace stlib
{
namespace geom
{

//
// Constructors
//

template<std::size_t N, typename T>
inline
void
SegmentMath<N, T>::
make(Point const& source, Point const& target)
{
  _segment = Simplex<T, N, 1>{{source, target}};
  _tangent = target;
  _tangent -= source;
  _length = ext::magnitude(_tangent);
  if (_length != 0) {
    _tangent /= _length;
  }
  else {
    _tangent[0] = 1;
  }
}


//
// Arithmetic member operators
//

template<std::size_t N, typename T>
inline
SegmentMath<N, T>&
SegmentMath<N, T>::
operator+=(Point const& p)
{
  _segment[0] += p;
  _segment[1] += p;
  return *this;
}


template<std::size_t N, typename T>
inline
SegmentMath<N, T>&
SegmentMath<N, T>::
operator-=(Point const& p)
{
  _segment[0] -= p;
  _segment[1] -= p;
  return *this;
}


//
// Binary free Operators
//


// Addition
template<std::size_t N, typename T>
inline
SegmentMath<N, T>
operator+(SegmentMath<N, T> s,
          const typename SegmentMath<N, T>::Point& p)
{
  s += p;
  return s;
}


// Subtraction
template<std::size_t N, typename T>
inline
SegmentMath<N, T>
operator-(SegmentMath<N, T> const& s,
          const typename SegmentMath<N, T>::Point& p)
{
  s -= p;
  return s;
}


//
// Mathematical member functions
//


template<std::size_t N, typename T>
inline
bool
SegmentMath<N, T>::
isValid() const
{
  Point tangent(getTarget() - getSource());
  ext::normalize(&tangent);
  return (ext::euclideanDistance(getTangent(), tangent) <
          10 * std::numeric_limits<Number>::epsilon() &&
          std::abs(ext::euclideanDistance(getSource(), getTarget()) -
                   getLength())
          < 10 * std::numeric_limits<Number>::epsilon());
}



//
// Mathematical Free Functions
//

template<std::size_t N, typename T>
inline
T
computeDistance(SegmentMath<N, T> const& segment,
                const typename SegmentMath<N, T>::Point& x)
{
  typename SegmentMath<N, T>::Point closestPoint;
  return computeDistanceAndClosestPoint(segment, x, &closestPoint);
}


template<std::size_t N, typename T>
inline
void
computeClosestPoint(SegmentMath<N, T> const& segment,
                    const typename SegmentMath<N, T>::Point& x,
                    typename SegmentMath<N, T>::Point* closestPoint)
{
  T proj = ext::dot(x - segment.getSource(), segment.getTangent());
  if (proj >= 0 && proj <= segment.getLength()) {
    *closestPoint = segment.getSource() + proj * segment.getTangent();
  }
  else {
    if (proj < 0) {
      *closestPoint = segment.getSource();
    }
    else {
      *closestPoint = segment.getTarget();
    }
  }
}


template<std::size_t N, typename T>
inline
T
computeDistanceAndClosestPoint
(SegmentMath<N, T> const& segment,
 const typename SegmentMath<N, T>::Point& x,
 typename SegmentMath<N, T>::Point* closestPoint)
{
  // Compute the closest point.
  computeClosestPoint(segment, x, closestPoint);
  // Return the distance.
  return ext::euclideanDistance(x, *closestPoint);
}


template<std::size_t N, typename T>
inline
T
computeUnsignedDistanceToSupportingLine
(SegmentMath<N, T> const& segment,
 const typename SegmentMath<N, T>::Point& x)
{
  typename SegmentMath<N, T>::Point closestPoint;
  return computeUnsignedDistanceAndClosestPointToSupportingLine
         (segment, x, &closestPoint);
}


template<std::size_t N, typename T>
inline
T
computeUnsignedDistanceAndClosestPointToSupportingLine
(SegmentMath<N, T> const& segment, const typename SegmentMath<N, T>::Point& x,
 typename SegmentMath<N, T>::Point* closestPoint)
{
  T const proj = ext::dot(x - segment.getSource(), segment.getTangent());
  *closestPoint = segment.getSource() + proj * segment.getTangent();
  return ext::euclideanDistance(x, *closestPoint);
}


template<typename T>
bool
computeZIntersection(const SegmentMath<3, T>& segment, T* x, T* y, T z)
{
  assert(segment.getSource()[2] <= segment.getTarget()[2]);

  // If the segment intersects the z plane.
  if (segment.getSource()[2] <= z && z <= segment.getTarget()[2]) {
    if (segment.getTangent()[2] > 1e-8) {
      const T a = (z - segment.getSource()[2]) / segment.getTangent()[2];
      *x = segment.getSource()[0] + a * segment.getTangent()[0];
      *y = segment.getSource()[1] + a * segment.getTangent()[1];
    }
    else {
      *x = segment.getSource()[0];
      *y = segment.getSource()[1];
    }
    return true;
  }
  return false;
}


template<typename T>
bool
computeIntersection(const SegmentMath<2, T>& s1, const SegmentMath<2, T>& s2,
                    typename SegmentMath<2, T>::Point* intersectionPoint)
{
  typedef typename SegmentMath<2, T>::Point Point;
  Point const& p1 = s1.getSource();
  Point const& p2 = s2.getSource();
  Point const& t1 = s1.getTangent();
  Point const& t2 = s2.getTangent();

  const T den = ext::discriminant(t1, t2);
  // If the segments are parallel.
  if (den == 0) {
    return false;
  }

  const T a = (t2[0] * (p1[1] - p2[1]) - t2[1] * (p1[0] - p2[0])) / den;
  *intersectionPoint = p1 + a * t1;
  const T b = ext::dot(*intersectionPoint - p2, t2);

  if (a >= 0 && a <= s1.getLength() && b >= 0 && b <= s2.getLength()) {
    return true;
  }

  return false;
}


//
// File IO
//


template<std::size_t N, typename T>
inline
std::istream&
operator>>(std::istream& in, SegmentMath<N, T>& s)
{
  typename SegmentMath<N, T>::Point source, target;
  in >> source >> target;
  s = SegmentMath<N, T>(source, target);
  return in;
}


template<std::size_t N, typename T>
inline
std::ostream&
operator<<(std::ostream& out, SegmentMath<N, T> const& s)
{
  return out << s.getSource() << "\n"
         << s.getTarget() << "\n"
         << s.getTangent() << "\n"
         << s.getLength() << "\n";
}


} // namespace geom
} // namespace stlib
