// -*- C++ -*-

/*!
  \file amr/writers.h
  \brief writers that stores data.
*/

#if !defined(__amr_writers_h__)
#define __amr_writers_h__

#include "stlib/amr/Orthtree.h"
#include "stlib/amr/PatchDescriptor.h"

#include <ostream>
#include <fstream>
#include <sstream>
#include <vector>

#include <sys/stat.h>

namespace stlib
{
namespace amr
{


//! Write the cell data in ParaView format.
/*!
  \relates Orthtree

  \param name The base name for the ParaView file and the VTK files.
  \param orthtree The orthtree.
  \param patchDescriptor The patch descriptor.
 */
template<typename _Patch, class _Traits>
void
writeCellDataParaview(const std::string& name,
                      const Orthtree<_Patch, _Traits>& orthtree,
                      const PatchDescriptor<_Traits>& patchDescriptor);


//! Write the bounding boxes for the patches in VTK format.
template<typename _Patch, class _Traits>
void
writePatchBoxesVtk(std::ostream& out, const Orthtree<_Patch, _Traits>& x);


} // namespace amr
}

#define __amr_writers_ipp__
#include "stlib/amr/writers.ipp"
#undef __amr_writers_ipp__

#endif
