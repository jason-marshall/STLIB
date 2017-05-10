// -*- C++ -*-

/**
  \file
  \brief Topology functions for simplices.
*/

#if !defined(__geom_kernel_simplexTopology_h__)
#define __geom_kernel_simplexTopology_h__

#include "stlib/ext/array.h"

namespace stlib
{
namespace geom
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//-----------------------------------------------------------------------------
/** \defgroup simplex_topology Topology Functions
*/
//@{

//---------------------------------------------------------------------------
// Simplex indices.
//---------------------------------------------------------------------------


/// Compute the other indices of the simplex.
void
computeOtherIndices(std::size_t i, std::size_t j, std::size_t* a,
                    std::size_t* b);


/// Compute the other index of the simplex.
std::size_t
computeOtherIndex(std::size_t i, std::size_t j, std::size_t k);


/// Reverse the orientation of the simplex.
template<typename _T, std::size_t N>
inline
void
reverseOrientation(std::array<_T, N>* simplex) {
  if (simplex->size() != 1) {
    std::swap((*simplex)[0], (*simplex)[1]);
  }
}

/// Get the indexed face obtained by removing the n_th vertex.
/**
  For an N-simplex, the face is (-1)^n (0, ..., n-1, n+1, ..., N).
*/
template<std::size_t N>
inline
std::array<std::size_t, N>
simplexIndexedFace(std::size_t const n) {
#ifdef STLIB_DEBUG
  assert(n <= N);
#endif
  std::array<std::size_t, N> face;
  for (std::size_t i = 0; i != n; ++i) {
    face[i] = i;
  }
  for (std::size_t i = n; i != N; ++i) {
    face[i] = i + 1;
  }
  if (n % 2 == 1) {
    reverseOrientation(&face);
  }
  return face;
}

/// Get the face obtained by removing the n_th vertex.
/**
  For the simplex (v[0], ... v[N]) the face is
  (-1)^n (v[0], ..., v[n-1], v[n+1], ..., v[N]).
*/
template<typename _T, std::size_t N>
inline
void
getFace(std::array<_T, N> const& simplex, std::size_t const n,
        std::array<_T, N - 1>* face) {
#ifdef STLIB_DEBUG
  assert(n < simplex.size());
#endif
  std::size_t j = 0;
  for (std::size_t i = 0; i != simplex.size(); ++i) {
    if (i != n) {
      (*face)[j++] = simplex[i];
    }
  }
  if (n % 2 == 1) {
    reverseOrientation(face);
  }
}

/// Return the face obtained by removing the n_th vertex.
/**
  For the simplex (v[0], ... v[N]) return
  (-1)^n (v[0], ..., v[n-1], v[n+1], ..., v[N]).
*/
template<typename _T, std::size_t N>
inline
std::array<_T, N - 1>
getFace(std::array<_T, N> const& simplex, std::size_t const n) {
  std::array<_T, N - 1> f;
  getFace(simplex, n, &f);
  return f;
}

/// Return true if the two simplices have the same orientation.
/**
  \pre \c x and \c y must have the same vertices.
*/
template<typename _T>
inline
bool
#ifdef STLIB_DEBUG
haveSameOrientation(std::array<_T, 0 + 1> const& x,
                    std::array<_T, 0 + 1> const& y) {
  assert(x[0] == y[0]);
  return true;
}
#else
haveSameOrientation(std::array<_T, 0 + 1> const& /*x*/,
                    std::array<_T, 0 + 1> const& /*y*/) {
  return true;
}
#endif

/// Return true if the two simplices have the same orientation.
/**
  \pre \c x and \c y must have the same vertices.
*/
template<typename _T>
inline
bool
haveSameOrientation(std::array<_T, 1 + 1> const& x,
                    std::array<_T, 1 + 1> const& y) {
#ifdef STLIB_DEBUG
  assert(hasElement(x, y[0]) && hasElement(x, y[1]));
#endif
  return x[0] == y[0];
}

/// Return true if the two simplices have the same orientation.
/**
  \pre \c x and \c y must have the same vertices.
*/
template<typename _T>
inline
bool
haveSameOrientation(std::array<_T, 2 + 1> const& x,
                    std::array<_T, 2 + 1> const& y) {
#ifdef STLIB_DEBUG
  assert(hasElement(x, y[0]) && hasElement(x, y[1]) && hasElement(x, y[2]));
#endif
  if (x[0] == y[0]) {
    return x[1] == y[1];
  }
  else if (x[0] == y[1]) {
    return x[1] == y[2];
  }
  // else x[0] == y[2]
  return x[1] == y[0];
}

/// Return true if the N-simplex has the specified (N-1)-face.
/**
  If true, set the face index.
*/
template<typename _T, std::size_t N, std::size_t _M>
inline
bool
hasFace(std::array<_T, N> const& simplex,
        std::array<_T, _M> const& face, std::size_t* faceIndex) {
  // Loop over the vertices of the face.
  for (std::size_t i = 0; i != face.size(); ++i) {
    // If the vertex of the face is not in the simplex.
    if (! ext::hasElement(simplex, face[i])) {
      *faceIndex = i;
      return false;
    }
  }
  // If we get here then all the vertices in the face are in the simplex.
  return true;
}

/// Return true if the simplex has the given face as a sub-simplex.
/**
  This function does not check the orientation of the face.  It returns
  true if the simplex has each of the vertices in the face.
*/
template<typename _T, std::size_t N, std::size_t _M>
inline
bool
hasFace(std::array<_T, N> const& simplex,
        std::array<_T, _M> const& face) {
  std::size_t faceIndex;
  return hasFace(simplex, face, &faceIndex);
}

/// Return true if the 3-simplex has the face specified by the three vertices.
template<typename _T>
inline
bool
hasFace(std::array<_T, 3 + 1> const& simplex,
        _T const& x, _T const& y, _T const& z) {
  return ext::hasElement(simplex, x) && ext::hasElement(simplex, y) &&
    ext::hasElement(simplex, z);
}

/// Return true if the 3-simplex has the face specified by the three vertices.
/**
  Set the face index.
*/
template<typename _T>
inline
bool
hasFace(std::array<_T, 3 + 1> const& simplex,
        _T const& x, _T const& y, _T const& z,
        std::size_t* faceIndex) {
  if (hasFace(simplex, x, y, z)) {
    *faceIndex = computeOtherIndex(ext::index(simplex, x),
                                   ext::index(simplex, y),
                                   ext::index(simplex, z));
    return true;
  }
  return false;
}

//@}

} // namespace geom
} // namespace stlib

#define __geom_kernel_simplexTopology_ipp__
#include "stlib/geom/kernel/simplexTopology.ipp"
#undef __geom_kernel_simplexTopology_ipp__

#endif
