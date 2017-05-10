// -*- C++ -*-

#include "stlib/concurrent/partition/BspTree.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/algorithm/statistics.h"
#include "stlib/ads/timer/Timer.h"

#include <iostream>
#include <fstream>
#include <sstream>

#include <cassert>

using namespace stlib;

namespace {

//
// Global variables.
//

//! The program name.
static std::string programName;

//
// Local functions.
//

//! Exit with an error message.
void
exitOnError() {
   std::cerr
         << "Bad arguments.  Usage:\n"
         << programName << " [-p] [-t=threshhold] cost.txt np\n"
         << "  Use -p to print the identifiers array.\n"
         << "  One can specify the thresshold for attempting to predict the best splitting.\n"
         << "  cost.txt is the file containing the cost array.\n"
         << "  np is the number of partitions.\n";
   exit(1);
}

}


//! The main loop.
int
main(int argc, char* argv[]) {
   typedef double Number;

   // The space dimension.
   const int Dimension = 2;

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the cost array and the number of partitions.
   if (parser.getNumberOfArguments() != 2) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the cost array.
   ads::Array<Dimension, Number> costs;
   {
      std::ifstream in(parser.getArgument().c_str());
      in >> costs;
      if (costs.empty()) {
         std::cerr << "Error: The cost array is empty.\n";
         exitOnError();
      }
   }

   int numberOfPartitions = 0;
   {
      std::istringstream in(parser.getArgument().c_str());
      in >> numberOfPartitions;
      if (numberOfPartitions < 1) {
         std::cerr << "Error: Bad value for the number of partitions: "
                   << numberOfPartitions << ".\n";
         exitOnError();
      }
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   const bool printIdentifiers = parser.getOption("p");

   int predictionThreshhold = 0;
   parser.getOption("t", &predictionThreshhold);

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Partition the elements.
   ads::Array<Dimension, int> identifiers(costs.extents());
   ads::Timer timer;
   timer.tic();
   concurrent::partitionRegularGridWithBspTree(costs, &identifiers,
         numberOfPartitions,
         predictionThreshhold);
   const double elapsedTime = timer.toc();

   if (printIdentifiers) {
      identifiers.pretty_print(std::cout);
   }

   std::cout << "Time to partition " << identifiers.size() << " elements into "
             << numberOfPartitions << " groups = " << elapsedTime
             << " seconds.\n";

   // Statistics on the elements per partition.
   {
      ads::Array<1, int> counts(numberOfPartitions, 0);
      for (int i = 0; i != identifiers.size(); ++i) {
         ++counts[identifiers[i]];
      }
      Number minimum, maximum, mean, sum;
      ads::computeMinimumMaximumAndMean
      (counts.begin(), counts.end(), &minimum, &maximum, &mean);
      sum = std::accumulate(counts.begin(), counts.end(), Number(0));
      std::cout << "Elements per partition:\n"
                << "  min = " << minimum
                << ", max = " << maximum
                << ", mean = " << mean
                << ", sum = " << sum << "\n";
   }

   // Compute the cost for each partition.
   ads::Array<1, Number> accumulatedCosts(numberOfPartitions, Number(0));
   for (int i = 0; i != costs.size(); ++i) {
      accumulatedCosts[identifiers[i]] += costs[i];
   }

   // Statistics on the cost per partition.
   {
      Number minimum, maximum, mean, sum;
      ads::computeMinimumMaximumAndMean
      (accumulatedCosts.begin(), accumulatedCosts.end(), &minimum, &maximum,
       &mean);
      sum = std::accumulate(accumulatedCosts.begin(), accumulatedCosts.end(),
                            Number(0));
      std::cout << "Cost per partition:\n"
                << "  min = " << minimum
                << ", max = " << maximum
                << ", mean = " << mean
                << ", sum = " << sum
                << ", efficiency = " << sum / maximum / numberOfPartitions
                << "\n";
   }

   // Communication statistics.
   {
      ads::Array<1, std::set<int> > neighbors(numberOfPartitions);
      ads::Array<1, int> sends(numberOfPartitions, 0);
      std::set<int> elementNeighbors;
      int id, n;
      const int extent0 = identifiers.extent(0);
      const int extent1 = identifiers.extent(1);
      for (int i = 0; i != extent0; ++i) {
         for (int j = 0; j != extent1; ++j) {
            elementNeighbors.clear();
            id = identifiers(i, j);

            if (i != 0) {
               n = identifiers(i - 1, j);
               if (n != id) {
                  neighbors[id].insert(n);
                  elementNeighbors.insert(n);
               }
            }

            if (j != 0) {
               n = identifiers(i, j - 1);
               if (n != id) {
                  neighbors[id].insert(n);
                  elementNeighbors.insert(n);
               }
            }

            if (i != extent0 - 1) {
               n = identifiers(i + 1, j);
               if (n != id) {
                  neighbors[id].insert(n);
                  elementNeighbors.insert(n);
               }
            }

            if (j != extent1 - 1) {
               n = identifiers(i, j + 1);
               if (n != id) {
                  neighbors[id].insert(n);
                  elementNeighbors.insert(n);
               }
            }

            sends[id] += elementNeighbors.size();
         }
      }

      ads::Array<1, int> numberOfNeighbors(numberOfPartitions);
      for (int i = 0; i != numberOfNeighbors.size(); ++i) {
         numberOfNeighbors = neighbors[i].size();
      }

      Number minimum, maximum, mean, sum;
      ads::computeMinimumMaximumAndMean
      (numberOfNeighbors.begin(), numberOfNeighbors.end(), &minimum, &maximum,
       &mean);
      sum = std::accumulate(numberOfNeighbors.begin(), numberOfNeighbors.end(),
                            Number(0));
      std::cout << "Adjacent neighbors for each partition:\n"
                << "  min = " << minimum
                << ", max = " << maximum
                << ", mean = " << mean
                << ", sum = " << sum << "\n";

      ads::computeMinimumMaximumAndMean
      (sends.begin(), sends.end(), &minimum, &maximum,
       &mean);
      sum = std::accumulate(sends.begin(), sends.end(), Number(0));
      std::cout << "Send operations for each partition:\n"
                << "  min = " << minimum
                << ", max = " << maximum
                << ", mean = " << mean
                << ", sum = " << sum << "\n";

      ads::Array<1, Number> commToComp(numberOfPartitions);
      commToComp = sends;
      commToComp /= accumulatedCosts;
      ads::computeMinimumMaximumAndMean
      (commToComp.begin(), commToComp.end(), &minimum, &maximum, &mean);
      std::cout << "Ratio of communication to computation:\n"
                << "  min = " << minimum
                << ", max = " << maximum
                << ", mean = " << mean << "\n";
   }

   return 0;
}
