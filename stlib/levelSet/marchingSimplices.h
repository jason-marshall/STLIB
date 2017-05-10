// -*- C++ -*-

#if !defined(__levelSet_marchingSimplices_h__)
#define __levelSet_marchingSimplices_h__

#include "stlib/levelSet/Grid.h"
#include "stlib/levelSet/components.h"
#include "stlib/levelSet/count.h"

#include "stlib/geom/kernel/content.h"
#include "stlib/ads/algorithm/sort.h"
#include "stlib/ads/iterator/TrivialOutputIterator.h"

namespace stlib
{
namespace levelSet
{


/*! \defgroup levelSetMarchingSimplices Marching Simplices
  The marching simplices algorithms for extracting an iso-surface from
  level set data. */
//@{


//! Return the content and boundary of the object defined by the level set.
/*! Output the triangle mesh of the boundary as a packed set of vertices. */
template<typename _T, std::size_t _D, std::size_t N, typename _OutputIterator>
void
contentAndBoundary(const Grid<_T, _D, N>& grid, _T* content, _T* boundary,
                   _OutputIterator vertices);


//! Return the content and boundary of the object defined by the level set.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
contentAndBoundary(const Grid<_T, _D, N>& grid, _T* content, _T* boundary)
{
  ads::TrivialOutputIterator vertices;
  contentAndBoundary(grid, content, boundary, vertices);
}


//! Return the content of the object defined by the level set.
template<typename _T, std::size_t _D, std::size_t N>
inline
_T
content(const Grid<_T, _D, N>& grid)
{
  _T c, b;
  ads::TrivialOutputIterator vertices;
  contentAndBoundary(grid, &c, &b, vertices);
  return c;
}


//! Return the boundary of the object defined by the level set.
template<typename _T, std::size_t _D, std::size_t N>
inline
_T
boundary(const Grid<_T, _D, N>& grid)
{
  _T c, b;
  ads::TrivialOutputIterator vertices;
  contentAndBoundary(grid, &c, &b, vertices);
  return b;
}


//! Return the content and boundary of each component of the object.
/*! The sizes of the content and boundary vectors will be set to the number
  of components. Output the triangle mesh of the boundary as a packed set
  of vertices. */
template<typename _T, std::size_t _D, std::size_t N, typename _OutputIterator>
void
contentAndBoundary(const Grid<_T, _D, N>& grid,
                   std::vector<_T>* content, std::vector<_T>* boundary,
                   _OutputIterator vertices);


//! Return the content and boundary of each component of the object.
/*! The sizes of the content and boundary vectors will be set to the number
  of components. */
template<typename _T, std::size_t _D, std::size_t N>
inline
void
contentAndBoundary(const Grid<_T, _D, N>& grid,
                   std::vector<_T>* content, std::vector<_T>* boundary)
{
  ads::TrivialOutputIterator vertices;
  contentAndBoundary(grid, content, boundary, vertices);
}

//@}


} // namespace levelSet
}

#define __levelSet_marchingSimplices_ipp__
#include "stlib/levelSet/marchingSimplices.ipp"
#undef __levelSet_marchingSimplices_ipp__

#endif
