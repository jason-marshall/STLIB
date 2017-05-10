// -*- C++ -*-

#include "stlib/geom/spatialIndexing/OrthtreeMap.h"

using namespace stlib;

int
main()
{
  {
    const std::size_t Dimension = 1;
    const std::size_t MaximumLevel = 10;
#define __test_geom_spatialIndexing_SpatialIndex_ipp__
#include "OrthtreeMap.ipp"
#undef __test_geom_spatialIndexing_SpatialIndex_ipp__
  }
  {
    const std::size_t Dimension = 2;
    const std::size_t MaximumLevel = 8;
#define __test_geom_spatialIndexing_SpatialIndex_ipp__
#include "OrthtreeMap.ipp"
#undef __test_geom_spatialIndexing_SpatialIndex_ipp__
  }
  {
    const std::size_t Dimension = 3;
    const std::size_t MaximumLevel = 6;
#define __test_geom_spatialIndexing_SpatialIndex_ipp__
#include "OrthtreeMap.ipp"
#undef __test_geom_spatialIndexing_SpatialIndex_ipp__
  }
  {
    const std::size_t Dimension = 4;
    const std::size_t MaximumLevel = 4;
#define __test_geom_spatialIndexing_SpatialIndex_ipp__
#include "OrthtreeMap.ipp"
#undef __test_geom_spatialIndexing_SpatialIndex_ipp__
  }

  return 0;
}
