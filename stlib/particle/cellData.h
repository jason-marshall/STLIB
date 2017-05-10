// -*- C++ -*-

/*!
  \file particle/cellData.h
  \brief CellData table for a sorted array of codes.
*/

#if !defined(__particle_cellData_h__)
#define __particle_cellData_h__

#include <algorithm>

#include <cstddef>

namespace stlib
{
namespace particle
{

//! Cell data. Used for performance analysis.
struct CellData {
  std::size_t occupancy;
  std::size_t sendCount;
  std::size_t process;
};

//! Return true if equal.
inline
bool
operator==(const CellData& x, const CellData& y)
{
  return x.occupancy == y.occupancy && x.sendCount == y.sendCount &&
         x.process == y.process;
}

//! Return true if not equal.
inline
bool
operator!=(const CellData& x, const CellData& y)
{
  return !(x == y);
}

//! Accumulate the fields.
inline
CellData&
operator+=(CellData& x, const CellData& y)
{
  // Add the occupancy counts.
  x.occupancy += y.occupancy;
  // For the send counts, choose the larger.
  x.sendCount = std::max(x.sendCount, y.sendCount);
  // For the process ID, choose the smaller.
  x.process = std::min(x.process, y.process);
  return x;
}

} // namespace particle
}

#endif
