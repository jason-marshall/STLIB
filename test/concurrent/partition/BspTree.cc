// -*- C++ -*-

#include "stlib/concurrent/partition/BspTree.h"

#include "stlib/ads/timer/Timer.h"
#include "stlib/container/MultiArray.h"

#include <iostream>

#include <cassert>

#define MI ext::make_array<std::ptrdiff_t>

typedef double Number;

void
test(const container::MultiArrayRef<Number, 1>& costs,
     container::MultiArrayRef<std::size_t, 1>* identifiers,
     const std::size_t numberOfPartitions) {
   concurrent::partitionRegularGridWithBspTree
   (costs, identifiers, numberOfPartitions);
   std::cout << "Costs =\n";
   for (std::size_t i = 0; i != costs.size(); ++i) {
      std::cout << costs[i] << " ";
   }
   std::cout << "\nNumber of partitions = " << numberOfPartitions << "\n"
             << "Identifiers =\n";
   for (std::size_t i = 0; i != identifiers->size(); ++i) {
      std::cout << (*identifiers)[i] << " ";
   }
   std::cout << "\n\n";
}

int
main(int argc, char* argv[]) {
   //==========================================================================
   // 1-D
   {
      const std::size_t Size = 10;
      container::MultiArray<Number, 1> costs(ext::make_array(Size));
      container::MultiArray<std::size_t, 1> identifiers(ext::make_array(Size));

      //------------------------------------------------------------------------
      // Constant cost.

      std::fill(costs.begin(), costs.end(), 1);
      std::size_t numberOfPartitions = 1;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 0 &&
             identifiers(MI(2)) == 0 &&
             identifiers(MI(3)) == 0 &&
             identifiers(MI(4)) == 0 &&
             identifiers(MI(5)) == 0 &&
             identifiers(MI(6)) == 0 &&
             identifiers(MI(7)) == 0 &&
             identifiers(MI(8)) == 0 &&
             identifiers(MI(9)) == 0);

      numberOfPartitions = 2;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 0 &&
             identifiers(MI(2)) == 0 &&
             identifiers(MI(3)) == 0 &&
             identifiers(MI(4)) == 0 &&
             identifiers(MI(5)) == 1 &&
             identifiers(MI(6)) == 1 &&
             identifiers(MI(7)) == 1 &&
             identifiers(MI(8)) == 1 &&
             identifiers(MI(9)) == 1);

      numberOfPartitions = 3;
      test(costs, &identifiers, numberOfPartitions);

      numberOfPartitions = 5;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 0 &&
             identifiers(MI(2)) == 1 &&
             identifiers(MI(3)) == 1 &&
             identifiers(MI(4)) == 2 &&
             identifiers(MI(5)) == 2 &&
             identifiers(MI(6)) == 3 &&
             identifiers(MI(7)) == 3 &&
             identifiers(MI(8)) == 4 &&
             identifiers(MI(9)) == 4);

      numberOfPartitions = 10;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 1 &&
             identifiers(MI(2)) == 2 &&
             identifiers(MI(3)) == 3 &&
             identifiers(MI(4)) == 4 &&
             identifiers(MI(5)) == 5 &&
             identifiers(MI(6)) == 6 &&
             identifiers(MI(7)) == 7 &&
             identifiers(MI(8)) == 8 &&
             identifiers(MI(9)) == 9);

      numberOfPartitions = 20;
      test(costs, &identifiers, numberOfPartitions);

      //------------------------------------------------------------------------
      // Different costs.  Ascending.

      for (std::size_t i = 0; i != costs.size(); ++i) {
         costs[i] = i + 1;
      }

      numberOfPartitions = 1;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 0 &&
             identifiers(MI(2)) == 0 &&
             identifiers(MI(3)) == 0 &&
             identifiers(MI(4)) == 0 &&
             identifiers(MI(5)) == 0 &&
             identifiers(MI(6)) == 0 &&
             identifiers(MI(7)) == 0 &&
             identifiers(MI(8)) == 0 &&
             identifiers(MI(9)) == 0);

      numberOfPartitions = 2;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 0 &&
             identifiers(MI(2)) == 0 &&
             identifiers(MI(3)) == 0 &&
             identifiers(MI(4)) == 0 &&
             identifiers(MI(5)) == 0 &&
             identifiers(MI(6)) == 0 &&
             identifiers(MI(7)) == 1 &&
             identifiers(MI(8)) == 1 &&
             identifiers(MI(9)) == 1);

      numberOfPartitions = 3;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 0 &&
             identifiers(MI(2)) == 0 &&
             identifiers(MI(3)) == 0 &&
             identifiers(MI(4)) == 0 &&
             identifiers(MI(5)) == 0 &&
             identifiers(MI(6)) == 1 &&
             identifiers(MI(7)) == 1 &&
             identifiers(MI(8)) == 2 &&
             identifiers(MI(9)) == 2);

      numberOfPartitions = 4;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 0 &&
             identifiers(MI(2)) == 0 &&
             identifiers(MI(3)) == 0 &&
             identifiers(MI(4)) == 0 &&
             identifiers(MI(5)) == 1 &&
             identifiers(MI(6)) == 1 &&
             identifiers(MI(7)) == 2 &&
             identifiers(MI(8)) == 2 &&
             identifiers(MI(9)) == 3);

      numberOfPartitions = 10;
      test(costs, &identifiers, numberOfPartitions);


      //------------------------------------------------------------------------
      // Different costs.  Descending.

      for (std::size_t i = 0; i != costs.size(); ++i) {
         costs[i] = costs.size() - i;
      }

      numberOfPartitions = 1;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 0 &&
             identifiers(MI(2)) == 0 &&
             identifiers(MI(3)) == 0 &&
             identifiers(MI(4)) == 0 &&
             identifiers(MI(5)) == 0 &&
             identifiers(MI(6)) == 0 &&
             identifiers(MI(7)) == 0 &&
             identifiers(MI(8)) == 0 &&
             identifiers(MI(9)) == 0);

      numberOfPartitions = 2;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 0 &&
             identifiers(MI(2)) == 0 &&
             identifiers(MI(3)) == 1 &&
             identifiers(MI(4)) == 1 &&
             identifiers(MI(5)) == 1 &&
             identifiers(MI(6)) == 1 &&
             identifiers(MI(7)) == 1 &&
             identifiers(MI(8)) == 1 &&
             identifiers(MI(9)) == 1);

      numberOfPartitions = 3;
      test(costs, &identifiers, numberOfPartitions);
      assert(identifiers(MI(0)) == 0 &&
             identifiers(MI(1)) == 0 &&
             identifiers(MI(2)) == 1 &&
             identifiers(MI(3)) == 1 &&
             identifiers(MI(4)) == 2 &&
             identifiers(MI(5)) == 2 &&
             identifiers(MI(6)) == 2 &&
             identifiers(MI(7)) == 2 &&
             identifiers(MI(8)) == 2 &&
             identifiers(MI(9)) == 2);
   }

   //==========================================================================
   // 2-D
   //------------------------------------------------------------------------
   // Constant cost.
   {
      const std::tr1::array<std::size_t, 2> Extents = {{4, 4}};
      container::MultiArray<Number, 2> costs(Extents);
      container::MultiArray<std::size_t, 2> ids(Extents);
      const std::size_t Partitions[] = {1, 2, 3, 4, 5, 8, 16, 100};
      std::fill(costs.begin(), costs.end(), 1);

      for (std::size_t i = 0; i != sizeof(Partitions) / sizeof(std::size_t);
      ++i) {
         concurrent::partitionRegularGridWithBspTree(costs, &ids, Partitions[i]);
         std::cout << "Costs =\n" << costs;
         //costs.pretty_print(std::cout);
         std::cout << "Number of partitions = " << Partitions[i] << "\n"
         << "Identifiers =\n" << ids;
         //ids.pretty_print(std::cout);
         std::cout << "\n";
      }
#if 0
      assert(ids(0, 0) == 0 && ids(1, 0) == 0 && ids(2, 0) == 0 && ids(3, 0) == 0 &&
      ids(0, 1) == 0 && ids(1, 1) == 0 && ids(2, 1) == 0 && ids(3, 1) == 0 &&
      ids(0, 2) == 0 && ids(1, 2) == 0 && ids(2, 2) == 0 && ids(3, 2) == 0 &&
      ids(0, 3) == 0 && ids(1, 3) == 0 && ids(2, 3) == 0 && ids(3, 3) == 0);
#endif
   }
   {
      const std::tr1::array<std::size_t, 2> Extents = {{16, 16}};
      container::MultiArray<Number, 2> costs(Extents);
      container::MultiArray<std::size_t, 2> ids(Extents);
      const std::size_t Partitions[] = {1, 2, 3, 4, 5, 8, 16, 256, 1000};
      std::fill(costs.begin(), costs.end(), 1);

      for (std::size_t i = 0; i != sizeof(Partitions) / sizeof(std::size_t);
      ++i) {
         concurrent::partitionRegularGridWithBspTree(costs, &ids, Partitions[i]);
         std::cout << "Costs =\n" << costs;
         //costs.pretty_print(std::cout);
         std::cout << "Number of partitions = " << Partitions[i] << "\n"
         << "Identifiers =\n" << ids;
         //ids.pretty_print(std::cout);
         std::cout << "\n";
      }
   }

   //==========================================================================
   // 3-D
   //------------------------------------------------------------------------
   // Constant cost.
   {
      const std::tr1::array<std::size_t, 3> Extents = {{4, 4, 4}};
      container::MultiArray<Number, 3> costs(Extents);
      container::MultiArray<std::size_t, 3> ids(Extents);
      const std::size_t Partitions[] = {1, 2, 3, 4, 5, 8, 16, 64, 100};
      std::fill(costs.begin(), costs.end(), 1);

      for (std::size_t i = 0; i != sizeof(Partitions) / sizeof(std::size_t);
      ++i) {
         concurrent::partitionRegularGridWithBspTree(costs, &ids, Partitions[i]);
         std::cout << "Costs =\n" << costs;
         //costs.pretty_print(std::cout);
         std::cout << "Number of partitions = " << Partitions[i] << "\n"
         << "Identifiers =\n" << ids;
         //ids.pretty_print(std::cout);
         std::cout << "\n";
      }
   }

   //==========================================================================
   // Aspect ratio, 2-D
   {
      std::size_t splittingIndex;
      double x, communication;

      ads::Timer timer;
      timer.tic();
      x = 1;
      for (double y = 1; y <= 2; ++y) {
         for (std::size_t n = 2; n <= 25; ++n) {
            communication = concurrent::predictBestSplitting(&splittingIndex,
            n, x, y);
            std::cout << "splittingIndex = " << splittingIndex
            << ", n = " << n
            << ", x = " << x
            << ", y = " << y
            << ", communication = " << communication << "\n";
         }
         std::cout << "\n";
      }
      ads::Timer::Number elapsedTime = timer.toc();
      std::cout << elapsedTime << "\n";
   }

   return 0;
}
