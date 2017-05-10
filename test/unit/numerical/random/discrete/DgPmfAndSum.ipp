// -*- C++ -*-

#ifndef __DgPmfAndSum_ipp__
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
   assert(f.sum() == std::accumulate(pmf.begin(), pmf.end(), 0.0));
   assert(f.isValid());

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

   f.print(std::cout);
}
