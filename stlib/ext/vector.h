// -*- C++ -*-

/**
  \file
  \brief Functions for vector.
*/

#if !defined(__ext_vector_h__)
#define __ext_vector_h__

#include "stlib/ext/array.h"

#include <vector>

#include <cstring>

/// Add using directives for the math operators for std::vector.
#define USING_STLIB_EXT_VECTOR_MATH_OPERATORS   \
  using stlib::ext::operator+=;                 \
  using stlib::ext::operator-=;                 \
  using stlib::ext::operator*=;                 \
  using stlib::ext::operator/=;                 \
  using stlib::ext::operator%=;                 \
  using stlib::ext::operator<<=;                \
  using stlib::ext::operator>>=

/// Add using directives for the input and output operators for std::vector.
#define USING_STLIB_EXT_VECTOR_IO_OPERATORS     \
  using stlib::ext::operator<<;                 \
  using stlib::ext::operator>>

/// Add using directives for the operators for std::vector.
#define USING_STLIB_EXT_VECTOR                  \
  USING_STLIB_EXT_VECTOR_MATH_OPERATORS;        \
  USING_STLIB_EXT_VECTOR_IO_OPERATORS

/**
\page extVector Extensions to std::vector

Here we provide functions to extend the functionality of the std::vector
class [\ref extAustern1999 "Austern, 1999"]. The functions are grouped into
the following categories:
- \ref extVectorAssignmentScalar
- \ref extVectorAssignmentVector
- \ref extVectorFile
- \ref extVectorMathematical

To use the operators, you will need to add using directives to your code.
You can either add them individually, or use one of the following convenience
macros.
\code
USING_STLIB_EXT_VECTOR_MATH_OPERATORS;
USING_STLIB_EXT_VECTOR_IO_OPERATORS;
USING_STLIB_EXT_VECTOR;
\endcode
*/

namespace stlib
{
namespace ext
{

//----------------------------------------------------------------------------
/// \defgroup extVectorAssignmentScalar Vector Assignment Operators with a Scalar Operand.
//@{

/// Vector-scalar addition.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator+=(std::vector<_T1>& x, const _T2& value)
{
  for (typename std::vector<_T1>::iterator i = x.begin(); i != x.end(); ++i) {
    *i += value;
  }
  return x;
}

/// Vector-scalar subtraction.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator-=(std::vector<_T1>& x, const _T2& value)
{
  for (typename std::vector<_T1>::iterator i = x.begin(); i != x.end(); ++i) {
    *i -= value;
  }
  return x;
}

/// Vector-scalar multiplication.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator*=(std::vector<_T1>& x, const _T2& value)
{
  for (typename std::vector<_T1>::iterator i = x.begin(); i != x.end(); ++i) {
    *i *= value;
  }
  return x;
}

/// Vector-scalar division.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator/=(std::vector<_T1>& x, const _T2& value)
{
#ifdef STLIB_DEBUG
  assert(value != 0);
#endif
  for (typename std::vector<_T1>::iterator i = x.begin(); i != x.end(); ++i) {
    *i /= value;
  }
  return x;
}

/// Vector-scalar modulus.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator%=(std::vector<_T1>& x, const _T2& value)
{
#ifdef STLIB_DEBUG
  assert(value != 0);
#endif
  for (typename std::vector<_T1>::iterator i = x.begin(); i != x.end(); ++i) {
    *i %= value;
  }
  return x;
}

/// Left shift.
template<typename _T>
inline
std::vector<_T>&
operator<<=(std::vector<_T>& x, const int offset)
{
  for (typename std::vector<_T>::iterator i = x.begin(); i != x.end(); ++i) {
    *i <<= offset;
  }
  return x;
}

/// Right shift.
template<typename _T>
inline
std::vector<_T>&
operator>>=(std::vector<_T>& x, const int offset)
{
  for (typename std::vector<_T>::iterator i = x.begin(); i != x.end(); ++i) {
    *i >>= offset;
  }
  return x;
}

//@}
//----------------------------------------------------------------------------
/// \defgroup extVectorAssignmentVector Vector Assignment Operators with a Vector Operand.
//@{


/// Vector-vector addition.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator+=(std::vector<_T1>& x, const std::vector<_T2>& y)
{
#ifdef STLIB_DEBUG
  assert(x.size() == y.size());
#endif
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] += y[n];
  }
  return x;
}

/// Vector-vector subtraction.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator-=(std::vector<_T1>& x, const std::vector<_T2>& y)
{
#ifdef STLIB_DEBUG
  assert(x.size() == y.size());
#endif
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] -= y[n];
  }
  return x;
}

/// Vector-vector multiplication.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator*=(std::vector<_T1>& x, const std::vector<_T2>& y)
{
#ifdef STLIB_DEBUG
  assert(x.size() == y.size());
#endif
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] *= y[n];
  }
  return x;
}

/// Vector-vector division.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator/=(std::vector<_T1>& x, const std::vector<_T2>& y)
{
#ifdef STLIB_DEBUG
  assert(x.size() == y.size());
#endif
  for (std::size_t n = 0; n != x.size(); ++n) {
#ifdef STLIB_DEBUG
    assert(y[n] != 0);
#endif
    x[n] /= y[n];
  }
  return x;
}

/// Vector-vector modulus.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator%=(std::vector<_T1>& x, const std::vector<_T2>& y)
{
#ifdef STLIB_DEBUG
  assert(x.size() == y.size());
#endif
  for (std::size_t n = 0; n != x.size(); ++n) {
#ifdef STLIB_DEBUG
    assert(y[n] != 0);
#endif
    x[n] %= y[n];
  }
  return x;
}

/// Vector-vector left shift.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator<<=(std::vector<_T1>& x, const std::vector<_T2>& y)
{
#ifdef STLIB_DEBUG
  assert(x.size() == y.size());
#endif
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] <<= y[n];
  }
  return x;
}

/// Vector-vector right shift.
template<typename _T1, typename _T2>
inline
std::vector<_T1>&
operator>>=(std::vector<_T1>& x, const std::vector<_T2>& y)
{
#ifdef STLIB_DEBUG
  assert(x.size() == y.size());
#endif
  for (std::size_t n = 0; n != x.size(); ++n) {
    x[n] >>= y[n];
  }
  return x;
}

//@}
//----------------------------------------------------------------------------
/// \defgroup extVectorFile Vector File I/O
//@{

/// Write the size and then the newline-separated elements.
/**
  Format:
  x.size()
  x[0]
  x[1]
  x[2]
  ...
*/
template<typename _T>
inline
std::ostream&
operator<<(std::ostream& out, const std::vector<_T>& x)
{
  out << x.size() << '\n';
  for (auto const& element: x) {
    out << element << '\n';
  }
  return out;
}

/// Read the size and then the elements.
/**
  The vector will be resized.
*/
template<typename _T>
inline
std::istream&
operator>>(std::istream& in, std::vector<_T>& x)
{
  std::size_t size;
  in >> size;
  x.resize(size);
  for (std::size_t n = 0; n != x.size(); ++n) {
    in >> x[n];
  }
  return in;
}

/// Write newline-separated elements.
/**
  Format:
  x[0]
  x[1]
  x[2]
  ...
*/
template<typename _T>
inline
void
writeElements(std::ostream& out, const std::vector<_T>& x)
{
  copy(x.begin(), x.end(), std::ostream_iterator<_T>(out, "\n"));
}

/// Read elements until the input is exhausted.
/**
  The vector will be resized.
*/
template<typename _T>
inline
void
readElements(std::istream& in, std::vector<_T>* x)
{
  x->clear();
  _T element;
  for (;;) {
    in >> element;
    // If the read was not successful, the stream is exhausted.
    if (! in) {
      break;
    }
    x->push_back(element);
  }
}

/// Write the size and then the elements in binary format.
template<typename _T>
inline
void
write(std::ostream& out, const std::vector<_T>& x)
{
  typedef typename std::vector<_T>::size_type size_type;
  // Write the size.
  size_type size = x.size();
  out.write(reinterpret_cast<const char*>(&size), sizeof(size_type));
  // Write the elements.
  out.write(reinterpret_cast<const char*>(&x[0]), x.size() * sizeof(_T));
}

/// Read the size and then the elements in binary format.
template<typename _T>
inline
void
read(std::istream& in, std::vector<_T>* x)
{
  typedef typename std::vector<_T>::size_type size_type;
  // Read the size.
  size_type size;
  in.read(reinterpret_cast<char*>(&size), sizeof(size_type));
  x->resize(size);
  // Read the elements.
  in.read(reinterpret_cast<char*>(&(*x)[0]), x->size() * sizeof(_T));
}

/// Return the size of the serialized data.
/** Use this when writing to a string. */
template<typename _T>
inline
std::size_t
serializedSize(const std::vector<_T>& x)
{
  return sizeof(typename std::vector<_T>::size_type) + sizeof(_T) * x.size();
}

/// Write the size and then the elements in binary format to the buffer.
template<typename _T>
inline
unsigned char*
write(unsigned char* out, const std::vector<_T>& x)
{
  typedef typename std::vector<_T>::size_type size_type;
  // Write the number of elements.
  size_type const size = x.size();
  memcpy(out, &size, sizeof(size_type));
  out += sizeof(size_type);
  // Write the elements.
  size_type const elementsSize = x.size() * sizeof(_T);
  memcpy(out, &x[0], elementsSize);
  out += elementsSize;
  return out;
}

/// Append the size and then the elements in binary format to the buffer.
/** Return the new size of the buffer. */
template<typename _T>
inline
std::size_t
write(std::vector<unsigned char>* buffer, const std::vector<_T>& x)
{
  unsigned char* p = 0;
  if (buffer->empty()) {
    buffer->resize(serializedSize(x));
    p = write(&(*buffer)[0], x);
  }
  else {
    // Copy the input state.
    std::vector<unsigned char> const oldBuffer = *buffer;
    // Allocate memory.
    buffer->resize(buffer->size() + serializedSize(x));
    // Copy in the original buffer.
    memcpy(&(*buffer)[0], &oldBuffer[0],
           oldBuffer.size() * sizeof(unsigned char));
    // Serialize the vector.
    p = write(&(*buffer)[oldBuffer.size()], x);
  }
  // Check that the sizes add up.
  if (p != &*buffer->end()) {
    throw std::runtime_error("Serialized data does not match the buffer "
                             "size.");
  }
  return buffer->size();
}

/// Read the size and then the elements in binary format from the buffer.
template<typename _T>
inline
unsigned char const*
read(unsigned char const* in, std::vector<_T>* x)
{
  typedef typename std::vector<_T>::size_type size_type;
  // Read the number of elements.
  size_type size;
  memcpy(&size, in, sizeof(size_type));
  in += sizeof(size_type);
  // Read the elements.
  x->resize(size);
  size_type const elementsSize = x->size() * sizeof(_T);
  memcpy(&x->front(), in, elementsSize);
  in += elementsSize;
  return in;
}

/// Read the size and then the elements in binary format from the buffer.
/** Return the position past the read portion of the buffer. */
template<typename _T>
inline
std::size_t
read(std::vector<unsigned char> const& buffer, std::vector<_T>* x,
     std::size_t const pos = 0)
{
  return std::distance(&buffer[pos], read(&buffer[pos], x));
}

//@}
//----------------------------------------------------------------------------
/// \defgroup extVectorMathematical Vector Mathematical Functions
//@{

/// Return the sum of the components.
template<typename _T>
inline
_T
sum(const std::vector<_T>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(0));
}

/// Return the product of the components.
template<typename _T>
inline
_T
product(const std::vector<_T>& x)
{
  return std::accumulate(x.begin(), x.end(), _T(1), std::multiplies<_T>());
}

/// Return the minimum component.  Use < for comparison.
template<typename _T>
inline
_T
min(const std::vector<_T>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::min_element(x.begin(), x.end());
}

/// Return the maximum component.  Use > for comparison.
template<typename _T>
inline
_T
max(const std::vector<_T>& x)
{
#ifdef STLIB_DEBUG
  assert(x.size() != 0);
#endif
  return *std::max_element(x.begin(), x.end());
}

/// Return the dot product of the two vectors.
template<typename _T1, typename _T2>
inline
_T1
dot(const std::vector<_T1>& x, const std::vector<_T2>& y)
{
  return std::inner_product(x.begin(), x.end(), y.begin(), _T1(0));
}

//@}

} // namespace ext
} // namespace stlib

#endif
