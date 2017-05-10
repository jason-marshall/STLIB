// -*- C++ -*-

#if !defined(__levelSet_contentFromDistance_h__)
#define __levelSet_contentFromDistance_h__

#include "stlib/numerical/constants.h"

#include <iterator>

#include <cmath>

namespace stlib
{
namespace levelSet
{


/*! \defgroup levelSetContentFromDistance Content from distance
  Compute the content of a manifold from its level set function specified
  as signed distance. */
//@{


//! Return the content of the manifold.
/*! The dimension must be specified explicitly. */
template<std::size_t _D, typename _InputIterator, typename _T>
_T
contentFromDistance(_InputIterator begin, _InputIterator end, _T dx);


//@}


} // namespace levelSet
}

#define __levelSet_contentFromDistance_ipp__
#include "stlib/levelSet/contentFromDistance.ipp"
#undef __levelSet_contentFromDistance_ipp__

#endif
