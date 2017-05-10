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

   Map map;
   std::vector<std::size_t> keys(count);
   for (std::size_t i = 0; i != keys.size(); ++i) {
      keys[i] = g();
      keys[i] <<= 32;
      keys[i] += g();
      map[keys[i]] = r();
   }
   
   ads::Timer timer;
   double result = 0;
   timer.tic();
   for (std::size_t i = 0; i != keys.size(); ++i) {
      result += map[keys[i]];
   }
   double elapsedTime = timer.toc();

   std::cout << "Meaningless result = " << result << '\n'
             << "Number of accesses = " << map.size() << '\n'
             << "Time per access = " << elapsedTime / map.size() * 1e9
             << " nanoseconds.\n";

   return 0;
}
