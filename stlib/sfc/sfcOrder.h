// -*- C++ -*-

#if !defined(stlib_sfc_sfcOrder_h)
#define stlib_sfc_sfcOrder_h

/**
  @file
  @brief Use a space-filling curve to determine an ordering for objects.
*/

#include "stlib/sfc/LocationCode.h"
#include "stlib/lorg/order.h"
#include "stlib/performance/Performance.h"

#include <utility>

namespace stlib
{
namespace sfc
{


/// Build a vector of ordered indices for the locations.
/**
  @param locations The vector of locations.
  @return The vector of ordered indices.
  @note The index type and the code type must be specified explicitly.
*/
template<typename Index, typename Code, typename Float, std::size_t D>
std::vector<Index>
sfcOrderSpecific(std::vector<std::array<Float, D> > const& locations);


/// Build a vector of ordered indices for the locations.
/**
  @param locations The vector of locations.
  @return The vector of ordered indices.
*/
template<typename Float, std::size_t D>
inline
std::vector<unsigned>
sfcOrder(std::vector<std::array<Float, D> > const& locations)
{
  if (locations.size() > unsigned(-1)) {
    throw std::runtime_error("In stlib::sfc::sfcOrder(): The number of "
                             "locations exceeds the capacity of the index "
                             "type.");
  }
  // CONTINUE Determine a good default for the code.
  return sfcOrderSpecific<unsigned, std::uint32_t>(locations);
}


/// Build a vector of ordered indices for the objects.
/**
  @param objects The vector of objects.
  @param location The function that returns a location for an object.
  @return The vector of ordered indices.
*/
template<typename Object, typename Location>
inline
std::vector<unsigned>
sfcOrder(std::vector<Object> const& objects,
         Location location)
{
  using Point = decltype(location(objects[0]));
  std::vector<Point> locations(objects.size());
  for (std::size_t i = 0; i != locations.size(); ++i) {
    locations[i] = location(objects[i]);
  }
  return sfcOrder(locations);
}


} // namespace sfc
} // namespace stlib

#define stlib_sfc_sfcOrder_tcc
#include "stlib/sfc/sfcOrder.tcc"
#undef stlib_sfc_sfcOrder_tcc

#endif
