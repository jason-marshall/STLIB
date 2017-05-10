// -*- C++ -*-

/**
  @file
  @brief Implements a class for an axis-aligned bounding box in N dimensions.
*/

#if !defined(__geom_BBox_h__)
#define __geom_BBox_h__

#include "stlib/ext/array.h"

#include <boost/config.hpp>

#include <vector>

#include <cmath>

namespace stlib
{
namespace geom
{

USING_STLIB_EXT_ARRAY;

/// An axis-oriented bounding box in the specified dimension.
/**
@param FloatT is the floating-point number type.
@param D is the dimension.



# Overview

A bounding box is used to provide a coarse description of the geometry
of an object, or set of objects. Thus, the most important operation 
is building -- one can build a bounding box that contains an object or
a range of objects. Also, one can add objects to an existing bounding box.
We have implemented building and adding to bounding boxes in a general
and extensible way. That is, there is built-in support for some 
common objects, like points, simplices, and bounding balls, and there
is a method for adding support for user-defined objects, like a polyhedron.

This class is a POD type. Thus, it has no user-defined constructors.
The box is defined by its lower and upper corners, which have the
data members @c lower and @c upper.
To construct a bounding box, use an initializer list. Below we construct
a bounding box that is the unit cube with lower corner at the origin.

@code{.cpp}
stlib::geom::BBox<float, 3> box = {{{0, 0, 0}}, {{1, 1, 1}}};
@endcode



# Default Bounding Boxes

Use the bbox() function, which is templated on the object type, 
to build bounding boxes. Note that you don't need to
specify the template parameters for the bounding box, i.e. the
floating-point number type and the space dimension, these will be
inferred from the object type that you are bounding. Specifically, the
BBoxForObject struct, which defines the @c DefaultBBox type, will be
used to determine an appropriate bounding box type for the object.  Of
course this class must be specialized for any object type that you
wish to bound.

Below we build an empty bounding box that is suitable for
bounding 3-D points using double-precision floating-point
numbers. We simply call the bbox() function with no arguments.
Note that we need to explicitly specify the template
parameter (the object type) because there are no function arguments.

@code{.cpp}
using Object = std::array<double, 3>;
auto box = stlib::geom::bbox<Object>();
assert(isEmpty(box));
@endcode

To build a bounding box for an object, use bbox() with a single
argument, namely the object being bounded.  Below are examples of
bounding various objects.

@code{.cpp}
using stlib::geom::bbox;
// Bound a point.
auto pointBBox = bbox(std::array<float, 2>{{2, 3}});
// Bound a bounding box.
stlib::geom::BBox<double, 2> square = {{{0, 0}}, {{1, 1}}};
auto boxBBox = bbox(square);
// Bound a triangle.
using Triangle = std::array<std::array<float, 2>, 3>;
Triangle triangle = {{{{0, 0}}, {{1, 0}}, {{0, 1}}}};
auto triangleBBox = bbox(triangle);
// Bound a bounding ball.
stlib::geom::Ball<float, 2> ball = {{{0, 0}}, 1};
auto ballBBox = bbox(ball);
@endcode

To build a bounding box for a range of objects, use
bbox() with two arguments, the iterators that define the range.
The value type for the iterators may be any supported object. Below we bound
a vector of tetrahedra (which has built-in support).

@code{.cpp}
using stlib::geom::bbox;
using Tetrahedron = std::array<std::array<double, 3>, 4>;
std::vector<Tetrahedron> tetrahedra;
// Add tetrahedra.
...
// Bound the tetrahedra.
auto domain = bbox(tetrahedra.begin(), tetrahedra.end());
@endcode



# Specific Bounding Boxes

Sometimes one must build bounding boxes that use one floating-point
number type for objects that use a different floating-point number
type.  For these cases. use specificBBox(). As with bbox(), the
function may take 0, 1, or 2 arguments. However, instead of being
templated on the object type, specificBBox() is templated on the
bounding box type. Note that the bounding box type must be specified
explicitly; it cannot be inferred from the arguments.

To construct an empty bounding box of a specific type, use the
specificBBox() function with no arguments. This may be useful when it
is convenient to specify the floating-point number type and the
dimension, but not the type of object that is being bounded. Below we
build an empty bounding box and verify that it is indeed empty.

@code{.cpp}
using BBox = stlib::geom::BBox<float, 2>;
auto const box = stlib::geom::specificBBox<BBox>();
assert(isEmpty(box));
@endcode

To bound an object, use specificBBox() with one argument. Bounding
the following objects is automatically supported.
- A Cartesian point, i.e. @c std::array<T, D>.
- A bounding box. Note that the number types need not be the same.
- An array of points.
- Ball, a bounding ball.
- A point/index pair. When passed a `std::pair`, we assume that `first` is
the geometric object to be bounded and `second` is the associated data.
.
Below are examples of bounding various objects.

@code{.cpp}
using stlib::geom::specificBBox;
using BBox = stlib::geom::BBox<float, 2>;
// Bound a point.
auto pointBBox = specificBBox<BBox>(std::array<float, 2>{{2, 3}});
// Bound a bounding box with a different number type.
stlib::geom::BBox<double, 2> square = {{{0, 0}}, {{1, 1}}};
auto boxBBox = specificBBox<BBox>(square);
// Bound a triangle.
using Triangle = std::array<std::array<float, 2>, 3>;
Triangle triangle = {{{{0, 0}}, {{1, 0}}, {{0, 1}}}};
auto triangleBBox = specificBBox<BBox>(triangle);
// Bound a bounding ball.
stlib::geom::Ball<float, 2> ball = {{{0, 0}}, 1};
auto ballBBox = specificBBox<BBox>(ball);
// Bound a point/index pair.
std::pair<std::array<float, 2>, std::size_t> pip = {{{2, 3}}, 7};
auto pipBBox = specificBBox<BBox>(pip);
@endcode

You can add support for more object types by specializing the BBoxForObject
struct. The static @c create() function specifies how to build the 
bounding box. Note that for the objects with built-in support, we 
allow conversions between number types. That is, we can build a 
single-precision bounding box for an object that uses double-precision
floating-point numbers, and vice versa.

To bound a range of objects, use specificBBox() with two arguments,
the iterators that define the range.  The value type for the iterators
may be any supported object. Below we bound a vector of tetrahedra
(which has built-in support).

@code{.cpp}
using stlib::geom::specificBBox;
using BBox = stlib::geom::BBox<double, 3>;
using Tetrahedron = std::array<std::array<double, 3>, 4>;
std::vector<Tetrahedron> tetrahedra;
// Add tetrahedra.
...
// Bound the tetrahedra.
BBox domain = specificBBox<BBox>(tetrahedra.begin(), tetrahedra.end());
@endcode

Consider an object type for which there is no built-in support for 
bounding. That is there is not already a specialization of BBoxForObject
defined for the type. Suppose that it is not possible to define such a 
specialization or that you don't feel like defining one. In such cases,
you can bound a range of objects by specifying a function that converts
the object to a boundable type. In the example below, the type of the 
objects in the range is @c std::size_t. The unary function, which we 
specify with a lambda expression, converts the index to a tetrahedron,
for which bounding is automatically supported.

@code{.cpp}
std::vector<std::size_t> indices;
// Select certain tetrahedra by recording their indices in the vector of tetrahedra.
...
// Bound the selected tetrahedra.
BBox domain = specificBBox<BBox>(indices.begin(), indices.end(),
                                 [&tetrahedra](std::size_t i){return tetrahedra[i];});
@endcode


# Operations and accessors

After building, the next most common thing that one does with bounding 
boxes is to add objects to them. That is, make the bounding box expand to 
contain the object. Objects are added to a bounding box with the += operator.
Below we start with an empty bounding box and add various objects to it.

@code{.cpp}
using BBox = stlib::geom::BBox<float, 2>;
// Start with an empty bounding box.
BBox box = specificBBox<BBox>();
// Add a point.
box += std::array<float, 2>{{2, 3}};
// Add a bounding box with a different number type.
box += stlib::geom::BBox<double, 2>{{{0, 0}}, {{1, 1}}};
// Add a triangle.
using Triangle = std::array<std::array<float, 2>, 3>;
box += Triangle{{{{0, 0}}, {{1, 0}}, {{0, 1}}}};
// Add a ball.
box += stlib::geom::Ball<float, 2>{{{0, 0}}, 1};
@endcode

You can check bounding boxes for equality and inequality. Note that 
any two empty bounding boxes are equal, regardless of the specific values
of the coordinates. In the following example, the lower and upper 
coordinates for the first bounding box are infinity and negative infinity,
respectively. The second bounding box has different coordinate values, 
but is also empty, so the two are equal.

@code{.cpp}
using BBox = stlib::geom::BBox<float, 2>;
assert(stlib::geom::specificBBox<BBox>() == (BBox{{{1, 1}}, {{0, 0}}}));
@endcode

You can read and write bounding boxes in ascii format. The representation of
the bounding box is just the lower and upper corners.

@code{.cpp}
using BBox = stlib::geom::BBox<float, 2>;
std::cout << "Enter a bounding box for the domain.\n";
std::cin >> domain;
std::cout << "The domain is " << domain << ".\n";
@endcode

You can calculate the centroid of a bounding box with
centroid(BBox<Float, D> const&).

@code{.cpp}
using BBox = stlib::geom::BBox<float, 2>;
assert(centroid(BBox{{{1, 2}}, {{5, 7}}}) == (std::array<float, 2>{{3, 4.5}}));
@endcode

Offsetting, expanding or contracting, is a common operation with bounding
boxes. Often one expands a bounding box to account for truncation errors
or to account for some interaction distance. In the following example
we expand the bounding box by the machine epsilon times the maximum edge 
length.

@code{.cpp}
using BBox = stlib::geom::BBox<double, 3>;
BBox domain = ...;
// Account for truncation errors that max occur in some subsequent calculations.
offset(&domain, std::numeric_limits<double>::epsilon() * max(domain.upper - domain.lower));
@endcode

Use isInside() to test whether an object is inside a bounding box.
Below are some examples.

@code{.cpp}
using stlib::geom::bbox;
using BBox = stlib::geom::BBox<double, 2>;
// No object is inside an empty bounding box.
assert(! isInside(bbox<BBox>(), std::array<double, 2>{{0, 0}}));
assert(! isInside(bbox<BBox>(), bbox<BBox>()));
// An empty bounding box is inside any non-empty bounding box.
assert(isInside(BBox{{{0, 0}}, {{0, 0}}}, bbox<BBox>()));
// The specified triangle in inside the unit square.
using Triangle = std::array<std::array<double, 2>, 3>;
Triangle triangle const = {{{{0, 0}}, {{1, 0}}, {{0, 1}}}};
assert(isInside(BBox{{{0, 0}}, {{1, 1}}}, triangle));
@endcode

Use content(BBox<Float, D> const&) to calculate the content
(length, area, volume, etc.) of a bounding box.

@code{.cpp}
using BBox = stlib::geom::BBox<double, 2>;
// The content of on empty bounding box is zero.
assert(content(stlib::geom::bbox<BBox>()) == 0);
// Check the area of a bounding box.
assert(content(BBox{{{0, 0}}, {{1, 2}}}) == 2);
@endcode

Use doOverlap(BBox<Float, D> const&, BBox<Float, D> const&) to check if two
bounding boxes overlap. Note that nothing overlaps an empty bounding box.

@code{.cpp}
using stlib::geom::bbox;
using BBox = stlib::geom::BBox<float, 1>;
assert(! doOverlap(bbox<BBox>(), bbox<BBox>()));
assert(! doOverlap(BBox{{{0}}, {{1}}}, bbox<BBox>()));
assert(doOverlap(BBox{{{0}}, {{1}}}, BBox{{{1}}, {{2}}}));
@endcode

To calculate the intersection of bounding boxes, use
intersection(BBox<Float, D> const&, BBox<Float, D> const&).

@code{.cpp}
using stlib::geom::bbox;
using BBox = stlib::geom::BBox<float, 1>;
assert(intersection(bbox<BBox>(), bbox<BBox>()) == bbox<BBox>());
assert(intersection(BBox{{{0}}, {{1}}}, BBox{{{1}}, {{2}}}) == (BBox{{{1}}, {{1}}}));
@endcode
*/
template<typename FloatT, std::size_t D>
struct BBox {
  // Types.

  /// The floating-point number type.
  using Float = FloatT;
  /// The point type.
  using Point = std::array<Float, D>;

  // Constants.

  /// The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = D;

  // Member data

  /// The lower corner.
  Point lower;
  /// The upper corner.
  Point upper;
};


/// Create an empty bounding box.
/** @relates BBox
    If any of the lower coordinates exceed the corresponding upper coordinate,
    then the bounding box is empty. Thus, for example, setting the lower 
    coordinates to 1 and the upper coordinates to 0 would yield an empty
    bounding box. However, in this function we go one step further. We set
    the coordinates to values that would likely cause an error if they were
    inadvertantly used. If the number type has a representation of infinity, 
    then the lower coordinates are set to positive infinity, while the 
    upper coordinates are set to negavite infinity. For other number types,
    we use use the @c max() and @c min() functions in @c std::numeric_limits
    to set coordinates values.
*/
template<typename BBoxT>
BBoxT
specificBBox();


/// Create a tight axis-aligned bounding box for the object.
/** Specialize this struct for every object that you want to bound
    with specificBBox(). The specialization must define the @c DefaultBBox
    type. It must also have a static create() function that takes a
    const reference to the object as an argument and returns the
    bounding box. */
template<typename Float, typename Object>
struct BBoxForObject;


/// Create a tight axis-aligned bounding box for the object.
/** @relates BBox */
template<typename BBoxT, typename Object>
inline
BBoxT
specificBBox(Object const& object)
{
  return BBoxForObject<typename BBoxT::Float, Object>::create(object);
}


/// Create a tight axis-aligned bounding box for the range of objects.
/** @relates BBox */
template<typename BBoxT, typename InputIterator>
BBoxT
specificBBox(InputIterator begin, InputIterator end);


/// Create a tight axis-aligned bounding box for the range of objects.
/** @relates BBox
    @param begin The beginning of a range of objects.
    @param end The end of a range of objects.
    @param boundable A unary function that when applied to an input object,
    returns an object that can be bounded by the unary specificBBox() function.
    For example, if the input objects were particles with positions, velocities,
    and masses, then the unary function would return the position. For another
    example, if the objects were polygons, then the unary function would 
    return a bounding box for the polygon argument.
*/
template<typename BBoxT, typename InputIterator, typename UnaryFunction>
BBoxT
specificBBox(InputIterator begin, InputIterator end, UnaryFunction boundable);


/// Create a tight axis-aligned bounding box for each object in the range.
/** @relates BBox */
template<typename BBoxT, typename ForwardIterator>
std::vector<BBoxT>
specificBBoxForEach(ForwardIterator begin, ForwardIterator end);


/// Create an empty bounding box.
/** @relates BBox
    If any of the lower coordinates exceed the corresponding upper coordinate,
    then the bounding box is empty. Thus, for example, setting the lower 
    coordinates to 1 and the upper coordinates to 0 would yield an empty
    bounding box. However, in this function we go one step further. We set
    the coordinates to values that would likely cause an error if they were
    inadvertantly used. If the number type has a representation of infinity, 
    then the lower coordinates are set to positive infinity, while the 
    upper coordinates are set to negavite infinity. For other number types,
    we use use the @c max() and @c min() functions in @c std::numeric_limits
    to set coordinates values.
*/
template<typename Object>
inline
typename BBoxForObject<void, Object>::DefaultBBox
bbox()
{
  return specificBBox<typename BBoxForObject<void, Object>::DefaultBBox>();
}


/// Create a tight axis-aligned bounding box for the object.
/** @relates BBox */
template<typename Object>
inline
typename BBoxForObject<void, Object>::DefaultBBox
bbox(Object const& object)
{
  return specificBBox<typename BBoxForObject<void, Object>::DefaultBBox>
    (object);
}


/// Create a tight axis-aligned bounding box for the range of objects.
/** @relates BBox */
template<typename InputIterator>
inline
typename BBoxForObject<void, typename std::iterator_traits<InputIterator>::value_type>::DefaultBBox
bbox(InputIterator begin, InputIterator end)
{
  using BBox = typename BBoxForObject<void, 
    typename std::iterator_traits<InputIterator>::value_type>::DefaultBBox;
  return specificBBox<BBox>(begin, end);
}


/// Create a tight axis-aligned bounding box for the range of objects.
/** @relates BBox
    @param begin The beginning of a range of objects.
    @param end The end of a range of objects.
    @param boundable A unary function that when applied to an input object,
    returns an object that can be bounded by the unary specificBBox() function.
    For example, if the input objects were particles with positions, velocities,
    and masses, then the unary function would return the position. For another
    example, if the objects were polygons, then the unary function would 
    return a bounding box for the polygon argument.
*/
template<typename InputIterator, typename UnaryFunction>
inline
typename BBoxForObject<void, typename std::iterator_traits<InputIterator>::value_type>::DefaultBBox
bbox(InputIterator begin, InputIterator end, UnaryFunction boundable)
{
  using BBox = typename BBoxForObject<void, 
    typename std::iterator_traits<InputIterator>::value_type>::DefaultBBox;
  return specificBBox<BBox>(begin, end, boundable);
}


/// Create a tight axis-aligned bounding box for each object in the range.
/** @relates BBox */
template<typename ForwardIterator>
inline
std::vector<typename BBoxForObject<void, typename std::iterator_traits<ForwardIterator>::value_type>::DefaultBBox>
defaultBBoxForEach(ForwardIterator begin, ForwardIterator end)
{
  using BBox = typename BBoxForObject<void, 
  typename std::iterator_traits<ForwardIterator>::value_type>::DefaultBBox;
  return specificBBoxForEach<BBox>(begin, end);
}


//
// Equality Operators.
//


/// Equality.
/** @relates BBox */
template<typename Float, std::size_t D>
inline
bool
operator==(BBox<Float, D> const& a, BBox<Float, D> const& b)
{
  return (isEmpty(a) && isEmpty(b)) ||
    (a.lower == b.lower && a.upper == b.upper);
}


/// Inequality.
/** @relates BBox */
template<typename Float, std::size_t D>
inline
bool
operator!=(BBox<Float, D> const& a, BBox<Float, D> const& b)
{
  return !(a == b);
}


//
// File I/O.
//


/// Read the bounding box.
/** @relates BBox */
template<typename Float, std::size_t D>
inline
std::istream&
operator>>(std::istream& in, BBox<Float, D>& x)
{
  return in >> x.lower >> x.upper;
}


/// Write the bounding box.
/** @relates BBox */
template<typename Float, std::size_t D>
inline
std::ostream&
operator<<(std::ostream& out, BBox<Float, D> const& x)
{
  return out << x.lower << ' ' << x.upper;
}


//
// Mathematical Functions.
//


/// Return true if the BBox is empty.
template<typename Float, std::size_t D>
bool
isEmpty(BBox<Float, D> const& box);


/// Return the centroid of the bounding box.
/** @note We do not check if the bounding box is empty. Empty bounding boxes
    do not have a centroid. Thus, the result is undefined. */
template<typename Float, std::size_t D>
inline
std::array<Float, D>
centroid(BBox<Float, D> const& box)
{
  return Float(0.5) * (box.lower + box.upper);
}


/// Offset (expand or contract) by the specified amount.
/** @relates BBox */
template<typename Float, std::size_t D>
inline
void
offset(BBox<Float, D>* box, Float const value)
{
  box->lower -= value;
  box->upper += value;
}


/// Make the bounding box expand to contain the object.
/** @relates BBox */
template<typename Float, std::size_t D, typename Object>
inline
BBox<Float, D>&
operator+=(BBox<Float, D>& box, Object const& object)
{
  return box += specificBBox<BBox<Float, D> >(object);
}


/// Make the bounding box expand to contain the point.
/** @relates BBox */
template<typename Float, std::size_t D>
BBox<Float, D>&
operator+=(BBox<Float, D>& box, std::array<Float, D> const& p);


/// Make the bounding box expand to contain the box.
/** @relates BBox */
template<typename Float, std::size_t D>
BBox<Float, D>&
operator+=(BBox<Float, D>& box, BBox<Float, D> const& rhs);


/// Return true if the object is contained in the bounding box.
/** @relates BBox */
template<typename Float, std::size_t D, typename Object>
bool
isInside(BBox<Float, D> const& box, Object const& x);


/// Return true if the point is in the bounding box.
/** @relates BBox */
template<typename Float, std::size_t D>
bool
isInside(BBox<Float, D> const& box, std::array<Float, D> const& p);


/// Return true if the second bounding box is in the first bounding box.
/** @relates BBox */
template<typename Float, std::size_t D>
inline
bool
isInside(BBox<Float, D> const& box, BBox<Float, D> const& x)
{
  // Nothing can be inside an empty bounding box.
  if (isEmpty(box)) {
    return false;
  }
  // Barring that, an empty bounding is inside any other bounding box.
  if (isEmpty(x)) {
    return true;
  }
  // If neither is empty, we compare coordinates.
  return isInside(box, x.lower) && isInside(box, x.upper);
}


/// Return the maximum absolute coordinate value.
/** @relates BBox
    This function is typically used when determining floating-point tolerances
    for performing arithmetic with the positions of the contained objects.
    @note The maximum absolute coordinate for an empty bounding box is zero
    because the box does not contain any objects.
*/
template<typename Float, std::size_t D>
inline
Float
maxAbsCoord(BBox<Float, D> const& box)
{
  if (isEmpty(box)) {
    return 0;
  }
  Float result = 0;
  for (std::size_t i = 0; i != D; ++i) {
    result = std::max(result, std::abs(box.lower[i]));
    result = std::max(result, std::abs(box.upper[i]));
  }
  return result;
}


/// Return the content (length, area, volume, etc.) of the box.
/** @relates BBox */
template<typename Float, std::size_t D>
inline
Float
content(BBox<Float, D> const& box)
{
  if (isEmpty(box)) {
    return 0;
  }
  Float x = 1;
  for (std::size_t n = 0; n != D; ++n) {
    x *= box.upper[n] - box.lower[n];
  }
  return x;
}


/// Return the squared distance between two 1-D intervals.
template<typename Float>
Float
squaredDistanceBetweenIntervals(Float lower1, Float upper1,
                                Float lower2, Float upper2);


/// Return the squared distance between two bounding boxes.
template<typename Float, std::size_t D>
Float
squaredDistance(BBox<Float, D> const& x, BBox<Float, D> const& y);


/// Return true if the bounding boxes overlap.
/** @relates BBox */
template<typename Float, std::size_t D>
bool
doOverlap(BBox<Float, D> const& a, BBox<Float, D> const& b);


/// Return the intersection of the bounding boxes.
/** @relates BBox */
template<typename Float, std::size_t D>
inline
BBox<Float, D>
intersection(BBox<Float, D> const& a, BBox<Float, D> const& b)
{
  return BBox<Float, D>{max(a.lower, b.lower), min(a.upper, b.upper)};
}




/// Scan convert the index bounding box.
/**
  @relates BBox

  @param indices is an output iterator for multi-indices. The value type
  must be std::array<Index,3> assignable where Index is the integer
  index type.
  @param box describes the range of indices. It is a bounding box of
  some floating point number type. This box will be converted to an
  integer bounding box. Then the below scan conversion function is used.
*/
template<typename Index, typename MultiIndexOutputIterator>
void
scanConvert(MultiIndexOutputIterator indices, BBox<double, 3> const& box);


/// Scan convert the index bounding box.
/**
  @relates BBox

  @param indices is an output iterator for multi-indices. The value type
  must be std::array<Index,3> assignable where Index is the integer
  index type.
  @param box describes the range of indices.
*/
template<typename MultiIndexOutputIterator, typename Index>
void
scanConvertIndex(MultiIndexOutputIterator indices, BBox<Index, 3> const& box);


/// Scan convert the index bounding box on the specified index domain.
/**
  @relates BBox

  @param indices is an output iterator for multi-indices. The value type
  must be std::array<Index,3> assignable where Index is the integer
  index type.
  @param box describes the range of indices. It is a bounding box of
  some floating point number type. This box will be converted to an
  integer bounding box. Then the below scan conversion function is used.
  @param domain is the closed range of indices on which to perform the
  scan conversion.
*/
template<typename MultiIndexOutputIterator, typename Index>
void
scanConvert(MultiIndexOutputIterator indices, BBox<double, 3> const& box,
            BBox<Index, 3> const& domain);


/// Scan convert the index bounding box on the specified index domain.
/**
  @relates BBox

  @param indices is an output iterator for multi-indices. The value type
  must be std::array<Index,3> assignable where Index is the integer
  index type.
  @param box is the closed range of indices.
  @param domain is the closed range of indices on which to perform the
  scan conversion.
*/
template<typename MultiIndexOutputIterator, typename Index>
inline
void
scanConvert(MultiIndexOutputIterator indices, BBox<Index, 3> const& box,
            BBox<Index, 3> const& domain);

} // namespace geom
} // namespace stlib

#define __geom_BBox_ipp__
#include "stlib/geom/kernel/BBox.ipp"
#undef __geom_BBox_ipp__

#endif
