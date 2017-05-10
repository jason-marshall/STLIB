// -*- C++ -*-

/**
  \file
  \brief A k-simplex is a sequence of k+1 vertices.
*/

#if !defined(__stlib_geom_kernel_Simplex_h__)
#define __stlib_geom_kernel_Simplex_h__

#include <array>

namespace stlib
{
namespace geom
{

/// A k-simplex in d-D space.
/**
   \note You may not be able to use this alias declaration as a parameter
   type in a function template because the arithmetic (_K + 1) may cause 
   template argument deduction to fail.
*/
template<typename _Float, std::size_t _D, std::size_t _K = _D>
using Simplex = std::array<std::array<_Float, _D>, _K + 1>;

/// A segment (1-simplex) in d-D space.
template<typename _Float, std::size_t _D>
using Segment = Simplex<_Float, _D, 1>;

/// A triangle (2-simplex) in d-D space.
template<typename _Float, std::size_t _D>
using Triangle = Simplex<_Float, _D, 2>;

/// A tetrahedron (3-simplex) in d-D space.
template<typename _Float, std::size_t _D>
using Tetrahedron = Simplex<_Float, _D, 3>;

} // namespace geom
} // namespace stlib

#endif
