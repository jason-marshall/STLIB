// -*- C++ -*-

#include "../kernel/Timer.h"

#include <iostream>
#include <strstream>
#include <vector>
#include <functional>

int
main(int argc, char* argv[]) {
   if (argc != 2) {
      std::cout << "Error: Bad arguments." << '\n'
                << "Usage: comp_int length" << '\n';
      exit(1);
   }

   //
   // Get the length of the vector.
   //
   istrstream length_str(argv[1]);
   size_t length;
   length_str >> length;

   //
   // Make a vector of random points.
   //
   vector< int > v(length);
   std::subtractive_rng random;
   typename vector<int>::iterator i = v.begin();
   typename vector<int>::const_iterator i_end = v.end();
   for (; i != i_end; ++i)
      *i = random(1000);

   int val = random(1000);

   size_t count(0);

   Timer timer;
   timer.tic();

   i = v.begin();
   while (i != i_end)
      if (*(i++) < val)
         ++count;

   double elapsed_time = timer.toc();

   std::cout << "Size = " << length << " count = " << count
             << " time = " << elapsed_time << '\n';

   return 0;
}
