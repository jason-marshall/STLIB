// -*- C++ -*-

#ifndef __RejectionBinsSplitting_ipp__
#error This is an implementation detail.
#endif

{
   std::vector<double> pmf(10, 1.0);

   Generator::DiscreteUniformGenerator uniform;
   Generator f(&uniform);
   f.initialize(pmf.begin(), pmf.end());

   {
      // Copy constructor.
      Generator g(f);
   }
   {
      // Assignment operator.
      Generator g(0);
      g = f;
   }
   {
      // Initialize.
      Generator g(&uniform);
      g.initialize(pmf.begin(), pmf.end());
      f.seed(1);
   }
   {
      // PMF constructor.
      Generator g(&uniform, pmf.begin(), pmf.end());
   }
   {
      // Seed.
      Generator g(&uniform);
      g.initialize(pmf.begin(), pmf.end());
      g.seed(1);
   }

   // Check the mean and variance.
   const std::size_t Size = 1000000;
   std::vector<double> data(Size);
   for (std::size_t n = 0; n != Size; ++n) {
      data[n] = f();
   }
   double mean, variance;
   ads::computeMeanAndVariance(data.begin(), data.end(), &mean, &variance);
   std::cout << "Size = " << Size << ", mean = " << mean
             << ", variance = " << variance << "\n";

   std::vector<std::size_t> counts(pmf.size(), 0);
   for (std::size_t n = 0; n != Size; ++n) {
      ++counts[f()];
   }
   std::cout << "Efficiency = " << f.computeEfficiency() << "\n";
   std::cout << "PMF:\n" << pmf << "Counts:\n" << counts << "\n";
   // Check that the counts are within 3 standard deviations of the mean.
   for (std::size_t i = 0; i != f.size(); ++i) {
      const double mean = Size * f[i] / f.sum();
      const double stdDev = std::sqrt(mean);
      assert(mean - 3 * stdDev <= counts[i] && counts[i] <= mean + 3 * stdDev);
   }
   //f.print(std::cout);

   // Test the dynamic capability.
   assert(f.sum() == sum(pmf));
   pmf[0] = 2.0;
   f.set(0, 2.0);
   assert(f.sum() == sum(pmf));

   // Use a different PMF.
   for (std::size_t i = 0; i != pmf.size(); ++i) {
      pmf[i] = i;
   }
   f.initialize(pmf.begin(), pmf.end());
   std::fill(counts.begin(), counts.end(), 0);
   for (std::size_t n = 0; n != Size; ++n) {
      ++counts[f()];
   }
   std::cout << "Efficiency = " << f.computeEfficiency() << "\n";
   std::cout << "PMF:\n" << pmf << "Counts:\n" << counts << "\n";
   // Check that the counts are within 3 standard deviations of the mean.
   for (std::size_t i = 0; i != f.size(); ++i) {
      const double mean = Size * f[i] / f.sum();
      const double stdDev = std::sqrt(mean);
      assert(mean - 3 * stdDev <= counts[i] && counts[i] <= mean + 3 * stdDev);
   }
   //f.print(std::cout);
}
