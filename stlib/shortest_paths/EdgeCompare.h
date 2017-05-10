// -*- C++ -*-

/*!
  \file EdgeCompare.h
  \brief Implements a class for comparing edges by weight.
*/

#if !defined(__EdgeCompare_h__)
#define __EdgeCompare_h__

#include <functional>

namespace stlib
{
namespace shortest_paths
{

//! Less than compare edges by weight.
template <typename EdgeType>
class EdgeCompare
  : public std::binary_function<EdgeType, EdgeType, bool>
{
public:
  //! Less than compare edges by weight.
  bool
  operator()(const EdgeType& x, const EdgeType& y) const
  {
    return (x.weight() < y.weight());
  }
};

//! Less than compare pointers to edges by weight.
template <typename EdgeType>
class EdgeCompare<EdgeType*>
  : public std::binary_function<EdgeType*, EdgeType*, bool>
{
public:
  //! Less than compare pointers to edges by weight.
  bool
  operator()(const EdgeType* x, const EdgeType* y) const
  {
    return (x->weight() < y->weight());
  }
};

//! Less than compare edges by source.
template <typename EdgeType>
class EdgeSourceCompare
  : public std::binary_function<EdgeType, EdgeType, bool>
{
public:
  //! Less than compare edges by source.
  bool
  operator()(const EdgeType& x, const EdgeType& y) const
  {
    return (x.source() < y.source());
  }
};

//! Less than compare edges by target.
template <typename EdgeType>
class EdgeTargetCompare
  : public std::binary_function<EdgeType, EdgeType, bool>
{
public:
  //! Less than compare edges by target.
  bool
  operator()(const EdgeType& x, const EdgeType& y) const
  {
    return (x.target() < y.target());
  }
};

} // namespace shortest_paths
}

#endif
