// -*- C++ -*-

/*!
  \file VertexCompare.h
  \brief Implements classes for comparing vertices by distance.
*/

#if !defined(__VertexCompare_h__)
#define __VertexCompare_h__

#include <functional>

namespace stlib
{
namespace shortest_paths
{

//! Less than compare vertices by distance.
template <typename VertexType>
class VertexCompare
  : public std::binary_function<VertexType, VertexType, bool>
{
public:
  //! Less than compare vertices by distance.
  bool
  operator()(const VertexType& x, const VertexType& y) const
  {
    return (x.distance() < y.distance());
  }
};

//! Greater than compare vertices by distance.
template <typename VertexType>
class VertexCompareGreater
  : public std::binary_function<VertexType, VertexType, bool>
{
public:
  //! Greater than compare vertices by distance.
  bool
  operator()(const VertexType& x, const VertexType& y) const
  {
    return (x.distance() > y.distance());
  }
};

//! Less than compare pointers to vertices by distance.
template <typename VertexType>
class VertexCompare<VertexType*>
  : public std::binary_function<VertexType*, VertexType*, bool>
{
public:
  //! Less than compare pointers to vertices by distance.
  bool
  operator()(const VertexType* x, const VertexType* y) const
  {
    return (x->distance() < y->distance());
  }
};

//! Greater than compare pointers to vertices by distance.
template <typename VertexType>
class VertexCompareGreater<VertexType*>
  : public std::binary_function<VertexType*, VertexType*, bool>
{
public:
  //! Greater than compare pointers to vertices by distance.
  bool
  operator()(const VertexType* x, const VertexType* y) const
  {
    return (x->distance() > y->distance());
  }
};

} // namespace shortest_paths
}

#endif
