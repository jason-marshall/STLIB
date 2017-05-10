// -*- C++ -*-

#ifndef __DgPmf_ipp__
#error This is an implementation detail.
#endif

{
   {
      // Default constructor.
      Pmf f;
      {
         // Copy constructor.
         Pmf g(f);
         assert(g == f);
      }
      {
         // Assignment operator.
         Pmf g;
         g = f;
         assert(g == f);
      }
   }

   //
   // Positive probabilities.
   //
   std::vector<double> pmf(10);
   for (std::size_t i = 0; i != pmf.size(); ++i) {
      pmf[i] = i + 1;
   }
   Pmf f;
   f.initialize(pmf.begin(), pmf.end());

   // Accessors.
   for (std::size_t i = 0; i != pmf.size(); ++i) {
      assert(f[i] == pmf[i]);
   }
   assert(f.size() == pmf.size());
   assert(std::equal(f.begin(), f.end(), pmf.begin()));

   // Manipulators.
   Pmf g;
   g.initialize(pmf.begin(), pmf.end());
   assert(g == f);
   for (std::size_t i = 0; i != g.size(); ++i) {
      g.set(i, 23);
   }
   for (std::size_t i = 0; i != g.size(); ++i) {
      assert(g[i] == 23);
   }
   std::copy(f.begin(), f.end(), g.begin());
   assert(g == f);

   f.print(std::cout);

   // CONTINUE Make sure this is tested elsewhere.
#if 0
   //
   // Non-negative probabilities.  0 and then 1.
   //
   pmf[0] = pmf[1] = pmf[2] = pmf[3] = pmf[4] = 0;
   pmf[5] = pmf[6] = pmf[7] = pmf[8] = pmf[9] = 1;
   f.initialize(pmf.begin(), pmf.end());
   // Random number generation.
   assert(f(0) == 0);
   assert(f(1) == 5);
   assert(f(2) == 6);
   assert(f(3) == 7);
   assert(f(4) == 8);
   assert(f(5) == 9);
   assert(f(6) == pmf.size() - 1);
   assert(f(1e10) == pmf.size() - 1);

   //
   // Non-negative probabilities.  1 and then 0.
   //
   pmf[0] = pmf[1] = pmf[2] = pmf[3] = pmf[4] = 1;
   pmf[5] = pmf[6] = pmf[7] = pmf[8] = pmf[9] = 0;
   f.initialize(pmf.begin(), pmf.end());
   // Random number generation.
   assert(f(0) == 0);
   assert(f(1) == 0);
   assert(f(2) == 1);
   assert(f(3) == 2);
   assert(f(4) == 3);
   assert(f(5) == 4);
   assert(f(6) == pmf.size() - 1);
   assert(f(1e10) == pmf.size() - 1);

   //
   // Non-negative probabilities.  Alternating 0 and 1.
   //
   pmf[0] = pmf[2] = pmf[4] = pmf[6] = pmf[8] = 0;
   pmf[1] = pmf[3] = pmf[5] = pmf[7] = pmf[9] = 1;
   f.initialize(pmf.begin(), pmf.end());
   // Random number generation.
   assert(f(0) == 0);
   assert(f(1) == 1);
   assert(f(2) == 3);
   assert(f(3) == 5);
   assert(f(4) == 7);
   assert(f(5) == 9);
   assert(f(6) == pmf.size() - 1);
   assert(f(1e10) == pmf.size() - 1);

   //
   // Non-negative probabilities.  Alternating 1 and 0.
   //
   pmf[0] = pmf[2] = pmf[4] = pmf[6] = pmf[8] = 1;
   pmf[1] = pmf[3] = pmf[5] = pmf[7] = pmf[9] = 0;
   f.initialize(pmf.begin(), pmf.end());
   // Random number generation.
   assert(f(0) == 0);
   assert(f(1) == 0);
   assert(f(2) == 2);
   assert(f(3) == 4);
   assert(f(4) == 6);
   assert(f(5) == 8);
   assert(f(6) == pmf.size() - 1);
   assert(f(1e10) == pmf.size() - 1);
#endif
}
