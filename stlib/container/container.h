// -*- C++ -*-

#if !defined(__container_all_h__)
#define __container_all_h__

#include "stlib/container/Array.h"
#include "stlib/container/EquilateralArray.h"
#include "stlib/container/MultiArray.h"
#include "stlib/container/PackedArrayOfArrays.h"
#include "stlib/container/SparseVector.h"
#include "stlib/container/StaticArrayOfArrays.h"
#include "stlib/container/SymmetricArray2D.h"
#include "stlib/container/TriangularArray.h"


/*!
  \file stlib/container/container.h
  \brief Includes the %container classes.
*/

namespace stlib
{
//! Namespace for container classes.
namespace container
{

/*!
\mainpage Containers

<!--------------------------------------------------------------------------->
\section container_introduction Introduction

This package provides classes for various containers.
All of the classes and functions are defined in the
container namespace. To use this package you can either include the header
for the class you are using
\code
#include "stlib/container/MultiArray.h"
\endcode
or you can include the convenience header.
\code
#include "stlib/container/container.h"
\endcode
(Here I assume that <tt>stlib/src</tt> is in your include path, i.e.
<tt>g++ -Istlib/src ...</tt>.)

<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection container_classes_array 1-D Dense Arrays

There are five array classes:
- container::Array allocates its memory and has contiguous storage.
- container::ArrayRef references memory and has contiguous storage.
- container::ArrayConstRef is the constant version of ArrayRef.
- container::ArrayView is a view of array data.
- container::ArrayConstView is the constant version of ArrayView.


<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection container_classes_multi Multidimensional Arrays

There are five multidimensional array classes:
- container::MultiArray allocates its memory and has contiguous storage.
- container::MultiArrayRef references memory and has contiguous storage.
- container::MultiArrayConstRef is the constant version of MultiArrayRef.
- container::MultiArrayView is a view of multi-dimensional array data.
- container::MultiArrayConstView is the constant version of MultiArrayView.

Each of these classes is templated on the value type and the dimension
(rank). For example MultiArray has the following declaration.
\code
template<typename _T, std::size_t _Dimension>
class MultiArray;
\endcode
The value type can be any class or built-in type. Below we construct a
3-dimensional array of integers.
\code
container::MultiArray<int, 3> a;
\endcode
Each of the multidimensional array classes have different constructors.
Consult the class documentation for details.

There are also three multidimensional array classes that have simpler
functionality:

- container::SimpleMultiArray allocates its memory.
- container::SimpleMultiArrayRef references memory.
- container::SimpleMultiArrayConstRef is the constant version of
SimpleMultiArrayRef.

<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection container_classes_other Other Array Classes

- container::EquilateralArray is a multi-array that has equal extents in
each dimension.
- container::SparseVector is a sparse 1-D vector. It stores index/value pairs.
- container::StaticArrayOfArrays is a static array of arrays (no, really).
It is an efficient way to represent static, sparse, 2-D arrays.
- container::PackedArrayOfArrays is like container::StaticArrayOfArrays,
but supports building on the fly by manipulating the last array.
- container::TriangularArray is a 2-D triangular array.
- container::SymmetricArray2D is a symmetric 2-D array with or without the
diagonal elements.

<!--CONTINUE HERE Update and move to a separate page.-->
<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -->
\subsection arrayArrayExamples Array Examples

We can construct a multidimensional array by specifying the index extents.
We use an array of <tt>std::size_t</tt> for the extents.
Below we define some types and construct a 3x4 2-D array.
\code
typedef container::MultiArray<double, 2> MultiArray;
typedef MultiArray::SizeList SizeList;
SizeList extents = {{3, 4}};
MultiArray a(extents);
\endcode
The analagous C multidimensional array would be declared as:
\code
double ca[3][4];
\endcode

We can treat 2-D array as simply a container of 12 elements. The
multidimensional array classes support the standard STL-style interface.
For those of you who are familiar with the C++ Standard Template Library (STL),
each of the multidimensional arrays classes fulfill the requirements of
a random access container. For the rest of you, I suggest you read
"Generic Programming and the STL" by Matthew H. Austern.
\code
assert(! a.empty());
assert(a.size() == 12);
assert(a.max_size() == 12);
std::fill(a.begin(), a.end(), 0);
for (std::size_t i = 0; i != a.size(); ++i) {
   a[i] = i;
}
\endcode
Note that <tt>operator[]</tt> performs container indexing and <em>not</em>
multidimensional array indexing.

The array classes use <tt>operator()</tt> to perform multidimensional
indexing. The argument is an array of integers. The index range for the
array we have declared is [0..2]x[0..3]. Below we define the
multidimensional index type and iterate over the elements and assign each
the sum of the indices.
\code
typedef MultiArray::IndexList IndexList;
IndexList i;
for (i[0] = 0; i[0] != 3; ++i[0]) {
  for (i[1] = 0; i[1] != 4; ++i[1]) {
    a(i) = sum(i);
  }
}
\endcode

The array we declared has indices that are zero-offset. That is, the lower
bounds for the index ranges are zero. We can also declare an array with
different index bases. The following array has index ranges [1..3]x[1..4].
\code
const SizeList extents = {{3, 4}};
const IndexList bases = {{1, 1}};
MultiArray b(extents, bases);
\endcode
The <tt>extents()</tt> and <tt>bases()</tt> accessors give the array extents
and the index bases.
\code
const IndexList lower = b.bases();
IndexList upper = lower;
upper += b.extents();
IndexList i;
for (i[0] = lower[0]; i[0] != upper[0]; ++i[0]) {
  for (i[1] = lower[0]; i[1] != upper[1]; ++i[1]) {
    b(i) = sum(i);
  }
}
\endcode




<!--------------------------------------------------------------------------->
\section arrayTypes Multidimensional Array Types

Each of the multidimensional array classes is an STL-compliant random access
containers. The following types are related to this functionality. First
the types defined in both the mutable and constant array classes.


- \c value_type is the element type of the array.
- \c const_pointer is a pointer to a constant array element.
- \c const_iterator is an iterator on constant elements in the array.
- \c const_reverse_iterator is a reverse iterator on constant elements in the
array.
- \c const_reference is a reference to a constant array element.
- \c size_type is the size type.
- \c difference_type is the pointer difference type.

The mutable array classes define mutable versions of the pointers, iterators,
and references.

- \c pointer is a pointer to an array element.
- \c iterator is an iterator on elements in the array.
- \c reverse_iterator is a reverse iterator on elements in the array.
- \c reference is a reference to an array element.

The remainder of the types start with a capital letter. They support
array indexing and working with different views of an array.

- \c Parameter is the parameter type. This is used for passing the value
type as an argument. If the value type is a built-in type then it is
\c value_type, otherwise it is <tt>const value_type&</tt>.
- \c Index is a single array index, which is a signed integer.
- \c IndexList is an array of indices:
<tt>std::array<std::ptrdiff_t, N></tt>. This type is
used for multidimensional array indexing.
- \c SizeList is an array of sizes: <tt>std::array<std:size_t, N></tt>.
This type is used to describe the index extents.
- \c Storage is a class that specifies the storage order.
- \c Range is a class that represents the index range for the array.
- \c ConstView is the class for a constant view of the array:
<tt>MultiArrayConstView<value_type, N></tt>.
- \c View is the class for a mutable view of the array:
<tt>MultiArrayView<value_type, N></tt>.




<!--------------------------------------------------------------------------->
\section arrayContainer The Multidimensional Array as a Random Access Container

The multidimensional array classes provide a number of member functions for
using the array as an STL random access container.

- \c empty() returns \c true if the array has zero elements.
- \c size() returns the number of elements.
- \c max_size() returns the number of elements as well. (Dynamically-sized
containers like \c std::vector return the maximum number of elements that
the container could hold, which is determined by the integer precision.
For statically-sized containers, the \c max_size() is the same as the
\c size().
- \c operator[]() returns the specified element in the array.
Below we sum the elements of an array and check it against the \c sum()
function.
\code
typedef MultiArray::value_type value_type;
value_type s = 0;
for (std::size_t i = 0; i != a.size(); ++i) {
  s += a[i];
}
assert(s == sum(a));
\endcode
Note again that \c operator[] performs container indexing and not
array indexing. In the following example we create a 1-D array with index
range [-5..5] but use container indexing to initialize the elements.
\code
typedef container::MultiArray<double, 1> MultiArray;
typedef MultiArray::SizeList SizeList;
typedef MultiArray::IndexList IndexList;
typedef MultiArray::size_type size_type;
const SizeList extents = {{11}};
const IndexList bases = {{-5}};
MultiArray a(extents, bases);
for (size_type i = 0; i != a.size(); ++i) {
  a[i] = i;
}
\endcode
- \c begin() returns a random access iterator to the first element.
- \c end() returns a random access iterator to one past the last element.
Below we copy the elements of an array \c a to a buffer.
\code
std::vector<double> buffer(a.size());
std::copy(a.begin(), a.end(), buffer.begin());
\endcode
- \c rbegin() returns a random access reverse iterator to the last element.
- \c rend()returns a random access reverse iterator to one past the first
element.
Below we copy the elements from \c a to \c b in reverse order.
\code
MultiArray b(a.extents());
assert(a.size() == b.size());
std::copy(a.rbegin(), a.rend(), b.begin());
\endcode
- \c fill() fills the array with the specified value. The following two
lines are equivalent.
\code
std::fill(a.begin(), a.end(), 1);
a.fill(1);
\endcode


<!--------------------------------------------------------------------------->
\section arrayIndexing Indexing operations.

With the array classes one may perform multidimensional array
indexing using the function call operator, \c operator(), with an
\c IndexList as an argument.
Below we create a 3-D array of integers
with index range [-5..5]x[-5..5]x[-5..5] and set the element values to the
product of the indices.
\code
typedef container::MultiArray<int, 3> MultiArray;
typedef MultiArray::SizeList SizeList;
typedef MultiArray::IndexList IndexList;
const SizeList extents = {{11, 11, 11}};
const IndexList bases = {{-5, -5, -5}};
MultiArray a(extents, bases);
const IndexList lower = a.bases();
IndexList upper = a.bases();
upper += a.extents();
IndexList i;
for (i[0] = lower[0]; i[0] != upper[0]; ++i[0]) {
  for (i[1] = lower[1]; i[1] != upper[1]; ++i[1]) {
    for (i[2] = lower[2]; i[2] != upper[2]; ++i[2]) {
      a(i) = product(i);
    }
  }
}
\endcode


For 1-D, 2-D, and 3-D arrays one may also use a list of indices to perform
the indexing. This is demonstrated below.
\code
typedef MultiArray::Index Index;
for (Index i = lower[0]; i != upper[0]; ++i) {
   for (Index j = lower[1]; j != upper[1]; ++j) {
      for (Index k = lower[2]; k != upper[2]; ++k) {
         a(i, j, k) = i * j * k;
      }
   }
}
\endcode

<!--------------------------------------------------------------------------->
\section arrayMultiIndexRange Index Ranges and Their Iterators

The MultiIndexRange class, as the name suggests, is used to represent index
ranges. We can describe a continuous index range by its extents and bases.
Consider an index range in 2-D. Below we construct the range [0..2]x[0..3]
\code
typedef container::MultiIndexRange<2> Range;
typedef Range::SizeList SizeList;
const SizeList extents = {{3, 4}};
Range range(extents);
\endcode
If the index range is not zero-offset, we can specify the index bases.
Below we construct the range [1..3]x[1..4].
\code
typedef Range::IndexList IndexList;
const SizeList extents = {{3, 4}};
const IndexList bases = {{1, 1}};
Range range(extents, bases);
\endcode

We can specify a non-continuous index range by specifying steps to take
in each dimension. Below we construct the index range
[1, 3, 5]x[1, 4, 7, 10].
\code
const SizeList extents = {{3, 4}};
const IndexList bases = {{1, 1}};
const IndexList steps = {{2, 3}};
Range range(extents, bases, steps);
\endcode

The \c range() member function returns the index range for each of the
the multidimensional array classes. Below we store the range for an array.
We check that the extents, bases, and steps are correct using the MultiIndexRange
accessor member functions.
\code
typedef container::MultiArray<double, 2> MultiArray;
typedef MultiArray::SizeList SizeList;
typedef MultiArray::IndexList IndexList;
typedef MultiArray::Range Range;
const SizeList extents = {{10, 20}};
MultiArray a(extents);
Range range = a.range();
assert(range.extents() == a.extents());
assert(range.bases() == a.bases());
assert(range.steps() == ext::filled_array<IndexList>(1));
\endcode

MultiIndexRange is not useful by itself. Rather, we use it to construct index
range iterators. MultiIndexRangeIterator is a random access iterator over an
index range. Below is a function that sets each element of an array to
the sum of its indices.
\code
template<std::size_t N>
void
setToSum(container::MultiArray<int, N>* a) {
  typedef container::MultiIndexRangeIterator<N> Iterator;
  const Iterator end = Iterator::end(a->range());
  for (Iterator i = Iterator::begin(a->range()); i != end; ++i) {
    (*a)(*i) = sum(*i);
  }
}
\endcode
We construct iterators to the beginning and end of the index range with
the static member functions \c begin() and \c end(). Dereferencing the
iterator yields an index.

Next we consider a more sophisticated example. Given a N-D arrays \c a
and \c b, we set the boundary elements of \c b to the same value
as in \c a, and set the interior elements
to the average of the adjacent neighbors in \c a. In 1-D,
<tt>b</tt><sub>i</sub> =
(<tt>a</tt><sub>i-1</sub> + <tt>a</tt><sub>i+1</sub>) / 2
for the interior elements. In 2-D we have
<tt>b</tt><sub>i,j</sub> =
(<tt>a</tt><sub>i-1,j</sub> + <tt>a</tt><sub>i+1,j</sub> +
<tt>a</tt><sub>i,j-1</sub> + <tt>a</tt><sub>i,j+1</sub>) / 4.
\code
template<typename _T, std::size_t N>
void
laplacianAverage(const container::MultiArray<_T, N>& a, container::MultiArray<_T, N>* b) {
  assert(a.range() == b->range());
  typedef container::MultiIndexRangeIterator<N> Iterator;
  typedef typename Iterator::IndexList IndexList;

  // Get the boundary values by copying the entire array.
  *b = a;

  // Skip if there are no interior points.
  if (min(a.extents()) <= 2) {
    return;
  }

  // The range for the interior elements.
  const container::MultiIndexRange<N> range(a.extents() - 2, a.bases() + 1);
  // Compute the interior values.
  _T s;
  IndexList index;
  const Iterator end = Iterator::end(range);
  for (Iterator i = Iterator::begin(range); i != end; ++i) {
    s = 0;
    for (std::size_t n = 0; n != N; ++n) {
      index = *i;
      index[n] -= 1;
      s += a(index);
      index[n] += 2;
      s += a(index);
    }
    (*b)(*i) = s / _T(2 * N);
  }
}
\endcode


<!--------------------------------------------------------------------------->
\section arrayReferences Multidimensional Array References

The MultiArrayRef and MultiArrayConstRef classes are multidimensional arrays
that reference externally allocated data. Their constructors differ from
MultiArray in that the first argument is a pointer to the data. Below is
a function that receives C arrays for the data, extents and bases and
constructs a 3-D MultiArrayRef.
\code
void
foo(double* data, const std::size_t extents[3], const int bases[3]) {
  typedef container::MultiArrayRef<double, 3> MultiArrayRef;
  typedef MultiArrayRef::SizeList SizeList;
  typedef MultiArrayRef::IndexList IndexList;
  MultiArrayRef a(data, ext::copy_array<SizeList>(extents),
                  ext::copy_array<IndexList>(bases));
  ...
}
\endcode
For MultiArrayConstRef the constructors take const pointers.

<!--------------------------------------------------------------------------->
\section arrayViews Multidimensional Array Views

MultiArrayView and MultiArrayConstView are array classes that can be used
to create views of existing arrays. The easiest way to construct them is to
use the \c view() member function with a specified index range. Below we
create a view of the interior elements of an array.
\code
typedef container::MultiArray<double, 2> MultiArray;
typedef MultiArray::SizeList SizeList;
typedef MultiArray::IndexList IndexList;
typedef MultiArray::Range Range;
typedef MultiArray::View View;
const SizeList extents = {{12, 12}};
const IndexList bases = {{-1, -1}};
MultiArray a(extents, bases);
const SizeList viewExtents = {{10, 10}};
View interior = a.view(Range(viewExtents));
\endcode

When using the \c view() function, the index bases of the array view are
the same as the index bases of the specified range. Below is a function
that returns a view of the interior elements of an array.
\code
template<typename _T, std::size_t N>
container::MultiArrayView<_T, N>
interior(container::MultiArray<_T, N>& a) {
  typedef typename container::MultiArray<_T, N>::Range Range;
  assert(min(a.extents()) > 2);
  return a.view(Range(a.extents() - 2, a.bases() + 1));
}
\endcode

The array view classes are fully fledged multidimensional arrays, just like
MultiArray, MultiArrayRef, and MultiArrayConstRef. In particular, they have
iterators. Below we use the \c interior() function be defined above and
set the interior elements of an array to zero.
\code
MultiArray a(extents, bases);
container::MultiArrayView<double, 2> b = interior(a);
typedef container::MultiArrayView<double, 2>::iterator iterator;
const iterator end = b.end();
for (iterator i = b.begin(); i != end; ++i) {
  *i = 0;
}
\endcode
These iterators are standard random access iterators. In addition, you can
get the index list that corresponds to the iterator position with the
\c indexList member function. Below we verify that this works correctly
for the \c interior array.
\code
for (iterator i = b.begin(); i != end; ++i) {
  assert(*i == b(i.indexList()));
}
\endcode

Note that the iterators for MultiArrayView and MultiArrayConstView are less
efficient than those for the other multidimensional array types. This is
because the data for the view classes is not contiguous in memory. The
view array classes use MultiViewIterator, while the rest of the classes use
raw pointers for iterators.

<!--------------------------------------------------------------------------->
\section arrayInheritance Inheritance

MultiArrayBase is a base class for the multidimensional arrays. It stores
the index extents and bases as well as the number of elements.

MultiArrayConstView derives from MultiArrayBase. It adds accessors
(like \c begin() and \c end()) that make
it a constant, random access container. It also adds constant array indexing
with \c operator().

MultiArrayView derives from MultiArrayConstView using virtual inheritance.
It adds the mutable interface.

MultiArrayConstRef derives from MultiArrayConstView using virtual inheritance.
It provides more efficient constant iterators and adds constant container
indexing.

MultiArrayRef derives from both MultiArrayConstRef and MultiArrayView.
It provides more
efficient mutable iterators and adds mutable container indexing.

MultiArray derives from MultiArrayRef. Other than the constructors
(and related functions), it hase the same interface as its base. It adds
memory allocation.

When writing functions which take these arrays as arguments, it is best to
use the least derived class which supplies the needed interface.
Consider the following function.
\code
template<typename _T, std::size_t N>
void
foo(container::MultiArray<_T, N>& a);
\endcode
Since the function only uses the constant interface, we can make the function
more general by accepting either a MultiArrayConstRef or a MultiArrayConstView.
If the function uses iterators on the array, then use the former because
arrays with contiguous data have more efficient iterators.
\code
template<typename _T, std::size_t N>
void
foo(container::MultiArrayConstRef<_T, N>& a);
\endcode
Otherwise use the latter.

Consider a function that modifies an array.
\code
template<typename _T, std::size_t N>
void
bar(container::MultiArray<_T, N>* a);
\endcode
We can make the function more general by accepting either a MultiArrayRef
or a MultiArrayView. Again the choice is usually based on whether we use
the array iterators. One needs to use a MultiArray as the argument type
only when changing the size of the array.


<!--------------------------------------------------------------------------->
\section container_other Other Multidimensional Array Libraries.

Why, oh why, did I write a multidimensional array package? Hasn't someone
already done that? Yes, there are a couple of good libraries. I wrote this
package because I wasn't entirely satisfied with other libraries.
Also, I was worried about depending on external libraries.
In writing this package I have stolen many ideas from Blitz++ and the
Boost multidimensional array library.

<a href="http://www.oonumerics.org/blitz/">Blitz++</a> is a full-featured
library and has good documentation. It is a powerful library that uses
template meta-programming to generate efficient code.
One thing that I don't like about Blitz++
is that it is not entirely STL-compliant and breaks some usual
C++ conventions. For instance, the copy
constructor does not copy an array, but references the data. (This is a feature
to keep naive programmers from shooting themselves in the foot.) Also, I've
had some compilation problems with the template expressions.

Multidimensional arrays are provided in the multi_array package of the
<a href="http://www.boost.org/">Boost</a> C++ libraries. It is a nice design
and has some cool tricks. However, in using this library I found it hard to
write dimension-independent code, i.e. write functions and classes where the
dimension is a template parameter.

*/

} // namespace container
}

#endif
