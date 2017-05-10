// -*- C++ -*-

#include "amr/DistributedOrthtree.h"
#include "amr/Patch.h"
#include "amr/PatchAdjacent.h"
#include "amr/Traits.h"

template<class _Orthtree>
void
printOrthtree(const _Orthtree& orthtree) {
   const int maxCount = 10;
   int count = 0;
   std::cout << "Orthtree with " << orthtree.size() << " nodes.\n";
   typename _Orthtree::const_iterator node = orthtree.begin();
   for (; node != orthtree.end() && count != maxCount; ++node, ++count) {
      // Print the spatial index and the first field of the first element in
      // the patch data.
      std::cout << node->first << " "
                << (*node->second.getPatchData().getArray().begin())[0];
      if (node->second.isGhost()) {
         std::cout << " ghost\n";
      }
      else {
         std::cout << "\n";
      }
   }
   if (node != orthtree.end()) {
      std::cout << "...\n";
   }
}

template<int Dimension, int MaximumLevel>
void
test(const int rank) {
   const int Depth = 1;
   const int GhostWidth = 1;

   typedef amr::Traits<Dimension, MaximumLevel> Traits;
   typedef typename Traits::SpatialIndex SpatialIndex;
   typedef typename Traits::Point Point;

   typedef amr::CellData<Traits, Depth, GhostWidth, int> CellData;
   typedef typename CellData::FieldTuple FieldTuple;
   typedef amr::Patch<CellData, Traits> Patch;
   typedef amr::PatchAdjacent<Patch, Traits> PatchAdjacent;
   typedef typename PatchAdjacent::SizeList SizeList;

   typedef amr::Orthtree<Patch, Traits> Orthtree;
   typedef typename Orthtree::iterator iterator;
   typedef typename Orthtree::const_iterator const_iterator;

   typedef amr::DistributedOrthtree<Patch, Traits, amr::PatchAdjacent>
   DistributedOrthtree;

   // Constructor.
   const Point lowerCorner = ext::filled_array<Point>(0.);
   const Point extents = ext::filled_array<Point>(1.);
   Orthtree orthtree(lowerCorner, extents);
   assert(orthtree.empty());
   // Key at level 0.
   const SpatialIndex key;
   PatchAdjacent helper(key, ext::filled_array<SizeList>(1), orthtree.end());
   assert(helper.getOrthtree().empty());
   DistributedOrthtree distributedOrthtree(MPI::COMM_WORLD, &helper);
   assert(distributedOrthtree.getOrthtree().empty());

   // Distribute from the specified process.
   if (MPI::COMM_WORLD.Get_rank() == rank) {
      std::cout << "\nDistribute from process " << rank << ".\n";
      helper.insert(SpatialIndex());
      helper.split(orthtree.begin());
      FieldTuple fieldTuple;
      int n = 0;
      for (iterator node = orthtree.begin(); node != orthtree.end(); ++node) {
         fieldTuple = n++;
         node->second.getPatchData().getArray().fill(fieldTuple);
      }
   }
   distributedOrthtree.partition();

   // Check that the processes got the correct nodes.
   {
      int numberOfNodes = 1 << Dimension;
      // Get this processes' partition of the nodes.
      int start, finish;
      numerical::getPartitionRange(numberOfNodes, MPI::COMM_WORLD.Get_size(),
                                   MPI::COMM_WORLD.Get_rank(), &start, &finish);
      assert(orthtree.size() == finish - start);
      FieldTuple fieldTuple;
      for (const_iterator node = orthtree.begin(); node != orthtree.end();
            ++node) {
         assert((*node->second.getPatchData().getArray().begin())[0] == start);
         ++start;
      }
   }
   // Check that each process knows that it owns its own nodes.
   for (const_iterator node = orthtree.begin(); node != orthtree.end(); ++node) {
      assert(distributedOrthtree.getProcess(node->first) ==
             MPI::COMM_WORLD.Get_rank());
   }

   distributedOrthtree.exchangeAdjacentSetUp();
   distributedOrthtree.exchangeAdjacent();
   if (MPI::COMM_WORLD.Get_rank() == 0) {
      printOrthtree(orthtree);
   }
   distributedOrthtree.exchangeAdjacentTearDown();

   const int balanceCount = distributedOrthtree.balance();
   if (MPI::COMM_WORLD.Get_rank() == 0) {
      std::cout << "Number of refinement operations for balancing = "
                << balanceCount << ".\n";
   }

   orthtree.clear();
}


template<int Dimension, int MaximumLevel>
void
test() {
   if (MPI::COMM_WORLD.Get_rank() == 0) {
      std::cout << "----------------------------------------------------------\n"
                << "Dimension = " << Dimension
                << ", MaximumLevel = " << MaximumLevel << "\n";
   }
   for (int rank = 0; rank != MPI::COMM_WORLD.Get_size(); ++rank) {
      test<Dimension, MaximumLevel>(rank);
   }
}


int
main(int argc, char* argv[]) {
   MPI::Init(argc, argv);

   test<1, 10>();
   test<2, 8>();
   test<3, 6>();
   test<4, 4>();

   MPI::Finalize();

   return 0;
}
