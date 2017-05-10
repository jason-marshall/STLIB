// -*- C++ -*-

/*!
  \file cuda/bound.h
  \brief Forward declarations for objects that can be bounded.
*/

#if !defined(__cuda_bound_h__)
#define __cuda_bound_h__

#include "stlib/ext/array.h"

namespace stlib
{
namespace geom
{

template<typename T, std::size_t N>
struct BBox;

// array

template<typename T, std::size_t N>
BBox<T, N>
bound(const std::array<T, N>& p);

template<typename T, std::size_t N>
BBox<T, N>
bound(const std::array<T, N>& p, const std::array<T, N>& q);

template<typename T, std::size_t N>
inline
BBox<T, N>
bound(const std::array<T, N>& p, const std::array<T, N>& q,
      const std::array<T, N>& r);

// BBox

template<typename T, std::size_t N>
const BBox<T, N>&
bound(const BBox<T, N>& x);


// Ball

template<typename T, std::size_t N>
class Ball;

template<typename T, std::size_t N>
BBox<T, N>
bound(const Ball<T, N>& ball);

} // namespace geom
}

#endif
