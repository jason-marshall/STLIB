// -*- C++ -*-

/*!
  \file cuda/vectorFloat3.h
  \brief Functions for the CUDA float3 vector type.
*/

#if !defined(__cuda_vectorFloat3_h__)
#define __cuda_vectorFloat3_h__

// CUDA includes.
#include <vector_functions.h>

#include <iostream>

//
// Equality.
//

//! Equality.
inline
__host__
__device__
bool
operator==(const float3 a, const float3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

//! Inequality.
inline
__host__
__device__
bool
operator!=(const float3 a, const float3 b)
{
  return !(a == b);
}

//
// Vector-scalar operations.
//

//! Vector-scalar addition.
inline
__host__
__device__
float3&
operator+=(float3& a, const float b)
{
  a.x += b;
  a.y += b;
  a.z += b;
  return a;
}

//! Vector-scalar subtraction.
inline
__host__
__device__
float3&
operator-=(float3& a, const float b)
{
  a.x -= b;
  a.y -= b;
  a.z -= b;
  return a;
}

//! Vector-scalar multiplication.
inline
__host__
__device__
float3&
operator*=(float3& a, const float b)
{
  a.x *= b;
  a.y *= b;
  a.z *= b;
  return a;
}

//! Vector-scalar division.
inline
__host__
__device__
float3&
operator/=(float3& a, const float b)
{
  a.x /= b;
  a.y /= b;
  a.z /= b;
  return a;
}

//
// Vector-vector operations.
//

//! Vector-vector addition.
inline
__host__
__device__
float3&
operator+=(float3& a, const float3 b)
{
  a.x += b.x;
  a.y += b.y;
  a.z += b.z;
  return a;
}

//! Vector-vector subtraction.
inline
__host__
__device__
float3&
operator-=(float3& a, const float3 b)
{
  a.x -= b.x;
  a.y -= b.y;
  a.z -= b.z;
  return a;
}

//! Vector-vector multiplication.
inline
__host__
__device__
float3&
operator*=(float3& a, const float3 b)
{
  a.x *= b.x;
  a.y *= b.y;
  a.z *= b.z;
  return a;
}

//! Vector-vector division.
inline
__host__
__device__
float3&
operator/=(float3& a, const float3 b)
{
  a.x /= b.x;
  a.y /= b.y;
  a.z /= b.z;
  return a;
}

//
// Unary operators.
//

//! Unary positive operator.
inline
__host__
__device__
const float3&
operator+(const float3& a)
{
  return a;
}

//! Unary negative operator.
inline
__host__
__device__
float3
operator-(float3 a)
{
  a.x = -a.x;
  a.y = -a.y;
  a.z = -a.z;
  return a;
}

//
// Binary operators.
//

//! Vector-scalar addition.
inline
__host__
__device__
float3
operator+(float3 a, const float b)
{
  a += b;
  return a;
}

//! Vector-scalar subtraction.
inline
__host__
__device__
float3
operator-(float3 a, const float b)
{
  a -= b;
  return a;
}

//! Vector-scalar multiplication.
inline
__host__
__device__
float3
operator*(float3 a, const float b)
{
  a *= b;
  return a;
}

//! Vector-scalar division.
inline
__host__
__device__
float3
operator/(float3 a, const float b)
{
  a /= b;
  return a;
}

//! Scalar-vector addition.
inline
__host__
__device__
float3
operator+(const float a, float3 b)
{
  b += a;
  return b;
}

//! Scalar-vector subtraction.
inline
__host__
__device__
float3
operator-(const float a, float3 b)
{
  b.x = a - b.x;
  b.y = a - b.y;
  b.z = a - b.z;
  return b;
}

//! Scalar-vector multiplication.
inline
__host__
__device__
float3
operator*(const float a, float3 b)
{
  b *= a;
  return b;
}

//! Scalar-vector division.
inline
__host__
__device__
float3
operator/(const float a, float3 b)
{
  b.x = a / b.x;
  b.y = a / b.y;
  b.z = a / b.z;
  return b;
}

//! Vector-vector addition.
inline
__host__
__device__
float3
operator+(float3 a, const float3 b)
{
  a += b;
  return a;
}

//! Vector-vector subtraction.
inline
__host__
__device__
float3
operator-(float3 a, const float3 b)
{
  a -= b;
  return a;
}

//! Vector-vector multiplication.
inline
__host__
__device__
float3
operator*(float3 a, const float3 b)
{
  a *= b;
  return a;
}

//! Vector-vector division.
inline
__host__
__device__
float3
operator/(float3 a, const float3 b)
{
  a /= b;
  return a;
}

//
// File I/O.
//

//! Write the space-separated elements.
inline
__host__
std::ostream&
operator<<(std::ostream& out, const float3& a)
{
  return out << a.x << ' ' << a.y << ' ' << a.z;
}

//! Read the elements.
inline
__host__
std::istream&
operator>>(std::istream& in, float3& a)
{
  return in >> a.x >> a.y >> a.z;
}

//! Write the elements in binary format.
inline
__host__
void
write(const float3& a, std::ostream& out)
{
  out.write(reinterpret_cast<const char*>(&a), sizeof(float3));
}

//! Read the elements in binary format.
inline
__host__
void
read(float3* a, std::istream& in)
{
  in.read(reinterpret_cast<char*>(a), sizeof(float3));
}

//
// Mathematical functions.
//

//! Return the sum of the components.
inline
__host__
__device__
float
sum(const float3& a)
{
  return a.x + a.y + a.z;
}

//! Return the product of the components.
inline
__host__
__device__
float
product(const float3& a)
{
  return a.x * a.y * a.z;
}

//! Return the minimum component.
inline
__host__
__device__
float
min(const float3& a)
{
  if (a.x < a.y) {
    if (a.x < a.z) {
      return a.x;
    }
    else {
      return a.z;
    }
  }
  else {
    if (a.y < a.z) {
      return a.y;
    }
    else {
      return a.z;
    }
  }
}

//! Return the maximum component.
inline
__host__
__device__
float
max(const float3& a)
{
  if (a.x > a.y) {
    if (a.x > a.z) {
      return a.x;
    }
    else {
      return a.z;
    }
  }
  else {
    if (a.y > a.z) {
      return a.y;
    }
    else {
      return a.z;
    }
  }
}

//! Return a vector that is element-wise the minimum of the two.
inline
__host__
__device__
float3
min(float3 a, const float3& b)
{
  a.x = min(a.x, b.x);
  a.y = min(a.y, b.y);
  a.z = min(a.z, b.z);
  return a;
}

//! Return a vector that is element-wise the maximum of the two.
inline
__host__
__device__
float3
max(float3 a, const float3& b)
{
  a.x = max(a.x, b.x);
  a.y = max(a.y, b.y);
  a.z = max(a.z, b.z);
  return a;
}

//! Return the dot product of the two vectors.
inline
__host__
__device__
float
dot(const float3& a, const float3& b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

//! Return the cross product of the two vectors.
inline
__host__
__device__
float3
cross(const float3& a, const float3& b)
{
  float3 c = {a.y* b.z - b.y * a.z,
              a.z* b.x - b.z * a.x,
              a.x* b.y - b.x * a.y
             };
  return c;
}

//! Return the triple product of the three vectors.
inline
__host__
__device__
float
tripleProduct(const float3& a, const float3& b, const float3& c)
{
  return dot(a, cross(b, c));
}

//! Return the squared magnitude.
inline
__host__
__device__
float
squaredMagnitude(const float3& a)
{
  return a.x * a.x + a.y * a.y + a.z * a.z;
}

//! Return the magnitude.
inline
__host__
__device__
float
magnitude(const float3& a)
{
  return sqrtf(squaredMagnitude(a));
}

//! Normalize the vector to have unit magnitude.
inline
__host__
__device__
void
normalize(float3* a)
{
  const float mag = magnitude(*a);
  if (mag != 0) {
    *a /= mag;
  }
  else {
    // If the vector has zero length, choose the unit vector whose first
    // coordinate is 1.
    a->x = 1;
    a->y = 0;
    a->z = 0;
  }
}

//! Return the squared distance between the two points.
inline
__host__
__device__
float
squaredDistance(const float3& a, const float3& b)
{
  return (a.x - b.x) * (a.x - b.x) +
         (a.y - b.y) * (a.y - b.y) +
         (a.z - b.z) * (a.z - b.z);
}

//! Return the Euclidean distance between the two points.
/*!
  \note distance would not be a good name for this function because
  std::distance() calculates the distance between iterators.
*/
inline
__host__
__device__
float
euclideanDistance(const float3& a, const float3& b)
{
  return sqrtf(squaredDistance(a, b));
}

#endif
