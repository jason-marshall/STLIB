// -*- C++ -*-

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

{
   std::vector<double> pmf(10, 1.0);

#ifdef USE_INFLUENCE
   container::StaticArrayOfArrays<std::size_t> influence;
   {
      // Independent probabilities.  Each probability only influences itself.
      std::vector<std::size_t> sizes(pmf.size()), values(pmf.size());
      std::fill(sizes.begin(), sizes.end(), 1);
      for (std::size_t i = 0; i != values.size(); ++i) {
         values[i] = i;
      }
      influence.rebuild(sizes.begin(), sizes.end(),
      values.begin(), values.end());
   }
#endif

   Generator::DiscreteUniformGenerator uniform;
   Generator f(&uniform);
#ifdef USE_INFLUENCE
   f.setInfluence(&influence);
#endif
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
      // PMF constructor.
#ifdef USE_INFLUENCE
      Generator g(&uniform, &influence, pmf.begin(), pmf.end());
#else
      Generator g(&uniform, pmf.begin(), pmf.end());
#endif
      // Seed.
      g.seed(1);
   }

   // Check the mean and variance.
   const std::size_t Size = 1000000;
   std::vector<double> data(Size);
   for (std::size_t n = 0; n != Size; ++n) {
      data[n] = f();
   }
   {
      double mean, variance;
      ads::computeMeanAndVariance(data.begin(), data.end(), &mean, &variance);
      std::cout << "Size = " << Size << ", mean = " << mean
      << ", variance = " << variance << "\n";
   }

   std::vector<std::size_t> counts(pmf.size(), std::size_t(0));
   for (std::size_t n = 0; n != Size; ++n) {
      ++counts[f()];
   }
   std::cout << "PMF:\n" << pmf << "\nCounts:\n" << counts << "\n";
   // Check that the counts are within 3 standard deviations of the mean.
   for (std::size_t i = 0; i != f.size(); ++i) {
      const double mean = Size * f[i] / f.sum();
      const double stdDev = std::sqrt(mean);
      assert(mean - 3 * stdDev <= counts[i] && counts[i] <= mean + 3 * stdDev);
   }

   // Use the PMF: 0, 1, 2, ..., 9.
   for (std::size_t i = 0; i != pmf.size(); ++i) {
      pmf[i] = i;
   }
   f.initialize(pmf.begin(), pmf.end());
   std::fill(counts.begin(), counts.end(), 0);
   for (std::size_t n = 0; n != Size; ++n) {
      ++counts[f()];
   }
   std::cout << "PMF:\n" << pmf << "\nCounts:\n" << counts << "\n";
   // Check that the counts are within 3 standard deviations of the mean.
   for (std::size_t i = 0; i != f.size(); ++i) {
      const double mean = Size * f[i] / f.sum();
      const double stdDev = std::sqrt(mean);
      assert(mean - 3 * stdDev <= counts[i] && counts[i] <= mean + 3 * stdDev);
   }

   // Use the PMF: 1, 0.1, 0.01, ..., 1e-9.
   for (std::size_t i = 0; i != pmf.size(); ++i) {
      pmf[i] = std::pow(10., -double(i));
   }
   f.initialize(pmf.begin(), pmf.end());
   std::fill(counts.begin(), counts.end(), 0);
   for (std::size_t n = 0; n != Size; ++n) {
      ++counts[f()];
   }
   std::cout << "PMF:\n" << pmf << "\nCounts:\n" << counts << "\n";
   // Check that the counts are within 3 standard deviations of the mean.
   for (std::size_t i = 0; i != f.size(); ++i) {
      const double mean = Size * f[i] / f.sum();
      std::cout << mean << ' ';
      //const double stdDev = std::sqrt(mean);
      //assert(mean - 3 * stdDev <= counts[i] && counts[i] <= mean + 3 * stdDev);
   }
   std::cout << '\n';
}
