// -*- C++ -*-

#include "geom/spatialIndexing/DistributedOrthtree.h"
#include "geom/spatialIndexing/OrthtreeMap.h"

int
main(int argc, char* argv[]) {
   MPI::Init(argc, argv);

   {
      const int Dimension = 1;
      const int MaximumLevel = 10;
#define __test_geom_spatialIndexing_DistributedOrthtree_ipp__
#include "DistributedOrthtree.ipp"
#undef __test_geom_spatialIndexing_DistributedOrthtree_ipp__
   }
   {
      const int Dimension = 2;
      const int MaximumLevel = 8;
#define __test_geom_spatialIndexing_DistributedOrthtree_ipp__
#include "DistributedOrthtree.ipp"
#undef __test_geom_spatialIndexing_DistributedOrthtree_ipp__
   }
   {
      const int Dimension = 3;
      const int MaximumLevel = 6;
#define __test_geom_spatialIndexing_DistributedOrthtree_ipp__
#include "DistributedOrthtree.ipp"
#undef __test_geom_spatialIndexing_DistributedOrthtree_ipp__
   }
   {
      const int Dimension = 4;
      const int MaximumLevel = 4;
#define __test_geom_spatialIndexing_DistributedOrthtree_ipp__
#include "DistributedOrthtree.ipp"
#undef __test_geom_spatialIndexing_DistributedOrthtree_ipp__
   }

   MPI::Finalize();

   return 0;
}
