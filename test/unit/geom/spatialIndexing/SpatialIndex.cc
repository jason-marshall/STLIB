// -*- C++ -*-

#include "stlib/geom/spatialIndexing/SpatialIndex.h"

using namespace stlib;

int
main()
{
  {
    const int Dimension = 1;
    const int MaximumLevel = 10;
#define __test_geom_spatialIndex_SpatialIndex_ipp__
#include "SpatialIndex.ipp"
#undef __test_geom_spatialIndex_SpatialIndex_ipp__
  }
  {
    const int Dimension = 2;
    const int MaximumLevel = 8;
#define __test_geom_spatialIndex_SpatialIndex_ipp__
#include "SpatialIndex.ipp"
#undef __test_geom_spatialIndex_SpatialIndex_ipp__
  }
  {
    const int Dimension = 3;
    const int MaximumLevel = 6;
#define __test_geom_spatialIndex_SpatialIndex_ipp__
#include "SpatialIndex.ipp"
#undef __test_geom_spatialIndex_SpatialIndex_ipp__
  }
  {
    const int Dimension = 4;
    const int MaximumLevel = 4;
#define __test_geom_spatialIndex_SpatialIndex_ipp__
#include "SpatialIndex.ipp"
#undef __test_geom_spatialIndex_SpatialIndex_ipp__
  }

  return 0;
}
