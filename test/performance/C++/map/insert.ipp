// -*- C++ -*-

#include "stlib/ads/timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/numerical/random/uniform.h"
#include "stlib/ext/pair.h"
#include "stlib/ext/vector.h"

#include <iostream>

using namespace stlib;

int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);

   std::size_t count = 0;
   parser.getArgument(&count);
   assert(count != 0);

   // Generator and uniform RNG.
   typedef numerical::ContinuousUniformGeneratorOpen<> Uniform;
   typedef Uniform::DiscreteUniformGenerator Generator;
   Generator g;
   Uniform r(&g);

   std::vector<std::pair<std::size_t, double> > values(count);
   for (std::size_t i = 0; i != count; ++i) {
      std::size_t key = g();
      key <<= 32;
      key += g();
      values[i] = std::make_pair(key, r());
   }
   
   Map map;
   ads::Timer timer;
   timer.tic();
   for (std::size_t i = 0; i != count; ++i) {
      map[values[i].first] = values[i].second;
   }
   double elapsedTime = timer.toc();

   std::cout << "Number of inserts = " << map.size() << '\n'
             << "Time per insert = " << elapsedTime / map.size() * 1e9
             << " nanoseconds.\n";

   return 0;
}
