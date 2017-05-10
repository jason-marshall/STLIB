// -*- C++ -*-

#if !defined(__levelSet_boolean_h__)
#define __levelSet_boolean_h__

#include "stlib/levelSet/Grid.h"
#include "stlib/container/SimpleMultiArray.h"
#include "stlib/ads/iterator/SingleValueIterator.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetBoolean Boolean Operations
\par
Let \e F and \e G denote be regions represented by the implicit functions
\e f and \e g. Below is a table of Boolean operations on the regions and the
corresponding operations on the implicit functions as well as the
library functions that implement the operation.
<table>
<tr> <th> Region <th> Implicit Function <th> Implementation
<tr> <td> complement(F) <td> <em>-f</em> <td> complement()
<tr> <td> \f$F \cap G\f$ <td> min(\e f, \e g) <td> unite()
<tr> <td> \f$F \cup G\f$ <td> max(\e f, \e g) <td> intersect()
<tr> <td> F - G <td> max(\e f, -\e g) <td> difference()
</table>
*/
//@{


//! Return true if the two implicit function values are equal.
template<typename _T>
bool
areFunctionValuesEqual(_T f, _T g);

//! Return true if the two regions are equal.
template<typename _InputIterator1, typename _InputIterator2>
bool
areFunctionsEqual(_InputIterator1 begin1, _InputIterator1 end1,
                  _InputIterator2 begin2);

//! Return true if the two regions are equal.
template<typename _T, std::size_t N>
bool
areFunctionsEqual(const container::SimpleMultiArrayConstRef<_T, N>& f,
                  const container::SimpleMultiArrayConstRef<_T, N>& g);

//! Return true if the two regions are equal.
template<typename _T, std::size_t _D, std::size_t N>
bool
areFunctionsEqual(const Grid<_T, _D, N>& f, const Grid<_T, _D, N>& g);


//! Calculate the complement of the region.
template<typename _ForwardIterator>
void
complement(_ForwardIterator begin, _ForwardIterator end);

//! Calculate the complement of the region.
template<typename _T, std::size_t N>
void
complement(container::SimpleMultiArrayRef<_T, N>* f);

//! Calculate the complement of the region.
template<typename _T, std::size_t _D, std::size_t N>
void
complement(Grid<_T, _D, N>* f);


//! For two implicit function values, calculate the value of the union.
/*! Check the special cases that either f or g are not known, i.e. are NaN. */
template<typename _T>
_T
unite(_T f, _T g);


//! For two implicit function values, calculate the value of the union.
/*! This wraps the unite() function into a functor. */
template<typename _T>
struct Unite :
    public std::binary_function<_T, _T, _T> {
  //! The base class.
  typedef std::binary_function<_T, _T, _T> Base;
  //! Calculate the union.
  typename Base::result_type
  operator()(typename Base::first_argument_type f,
             typename Base::second_argument_type g)
  {
    return unite(f, g);
  }
};


//! Calculate the union of the two regions.
template<typename _T, std::size_t N>
void
unite(container::SimpleMultiArrayRef<_T, N>* f,
      const container::SimpleMultiArrayConstRef<_T, N>& g);


//! Calculate the union of the two regions.
/*!
  \pre The grids must have the same extents. The output grid may not be
  either input grid.
*/
template<typename _T, std::size_t _D, std::size_t N>
void
unite(const Grid<_T, _D, N>& f, const Grid<_T, _D, N>& g,
      Grid<_T, _D, N>* result);

//! For two implicit function values, calculate the value of the intersection.
/*! Check the special cases that either f or g are not known, i.e. are NaN. */
template<typename _T>
_T
intersect(_T f, _T g);


//! For two implicit function values, calculate the value of the intersection.
/*! This wraps the intersect() function into a functor. */
template<typename _T>
struct Intersect :
    public std::binary_function<_T, _T, _T> {
  //! The base class.
  typedef std::binary_function<_T, _T, _T> Base;
  //! Calculate the union.
  typename Base::result_type
  operator()(typename Base::first_argument_type f,
             typename Base::second_argument_type g)
  {
    return intersect(f, g);
  }
};


//! Calculate the intersection of the two regions.
template<typename _T, std::size_t N>
void
intersect(container::SimpleMultiArrayRef<_T, N>* f,
          const container::SimpleMultiArrayConstRef<_T, N>& g);


//! Calculate the intersection of the two regions.
template<typename _T, std::size_t _D, std::size_t N>
void
intersect(const Grid<_T, _D, N>& f, const Grid<_T, _D, N>& g,
          Grid<_T, _D, N>* result);

//! For two implicit function values, calculate the boolean difference.
/*! Check the special cases that either f or g are not known, i.e. are NaN. */
template<typename _T>
_T
difference(_T f, _T g);


//! For two implicit function values, calculate the value of the difference.
/*! This wraps the difference() function into a functor. */
template<typename _T>
struct Difference :
    public std::binary_function<_T, _T, _T> {
  //! The base class.
  typedef std::binary_function<_T, _T, _T> Base;
  //! Calculate the union.
  typename Base::result_type
  operator()(typename Base::first_argument_type f,
             typename Base::second_argument_type g)
  {
    return difference(f, g);
  }
};


//! Calculate the difference of the two regions.
template<typename _T, std::size_t N>
void
difference(container::SimpleMultiArrayRef<_T, N>* f,
           const container::SimpleMultiArrayConstRef<_T, N>& g);


//! Calculate the difference of the two regions.
template<typename _T, std::size_t _D, std::size_t N>
void
difference(const Grid<_T, _D, N>& f, const Grid<_T, _D, N>& g,
           Grid<_T, _D, N>* result);


//@}

} // namespace levelSet
}

#define __levelSet_boolean_ipp__
#include "stlib/levelSet/boolean.ipp"
#undef __levelSet_boolean_ipp__

#endif
