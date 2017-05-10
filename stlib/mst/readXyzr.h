// -*- C++ -*-

/*!
  \file readXyzr.h
  \brief Read an xyzr protein file.
*/

#if !defined(__mst_readXyzr_h__)
#define __mst_readXyzr_h__

#include "stlib/mst/Atom.h"

#include <iostream>
#include <fstream>

#include <cassert>

namespace stlib
{
namespace mst
{

USING_STLIB_EXT_ARRAY;

//! Given a file name, read a molecule in xyzr format.
template < typename T, typename PointOutputIterator,
           typename NumberOutputIterator >
void
readXyzr(const char* fileName, PointOutputIterator points,
         NumberOutputIterator radii);


//! Given an input stream, read a molecule in xyzr format.
template < typename T, typename PointOutputIterator,
           typename NumberOutputIterator >
void
readXyzr(std::istream& inputStream, PointOutputIterator points,
         NumberOutputIterator radii);



//! Given a file name, read a molecule in xyzr format.
template<typename T, typename AtomOutputIterator>
void
readXyzr(const char* fileName, AtomOutputIterator atoms);


//! Given an input stream, read a molecule in xyzr format.
template<typename T, typename AtomOutputIterator>
void
readXyzr(std::istream& inputStream, AtomOutputIterator atoms);


} // namespace mst
}

#define __mst_readXyzr_ipp__
#include "stlib/mst/readXyzr.ipp"
#undef __mst_readXyzr_ipp__

#endif
