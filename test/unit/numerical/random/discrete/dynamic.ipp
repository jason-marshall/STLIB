// -*- C++ -*-

{
   const std::size_t Size = 10;
#ifdef USE_INFLUENCE
   container::StaticArrayOfArrays<std::size_t> influence;
   {
      // Independent probabilities.  Each probability only influences itself.
      std::vector<std::size_t> sizes(Size), values(Size);
      std::fill(sizes.begin(), sizes.end(), 1);
      for (std::size_t i = 0; i != values.size(); ++i) {
         values[i] = i;
      }
      influence.rebuild(sizes.begin(), sizes.end(), values.begin(), values.end());
   }
#endif

   Generator::DiscreteUniformGenerator uniform;
   Generator f(&uniform);
#ifdef USE_INFLUENCE
   f.setInfluence(&influence);
#endif
   {
      // Validity.
      std::vector<double> pmf(Size, 1.);
      f.initialize(pmf.begin(), pmf.end());
      assert(f.isValid());
      for (std::size_t i = 0; i != pmf.size(); ++i) {
         f.set(i, 0.);
      }
      updateSum(&f);
      assert(! f.isValid());
   }
   {
      // Constant.
      std::vector<double> pmf(Size, 10.);
      const std::size_t Sum = 
         static_cast<std::size_t>(std::accumulate(pmf.begin(), pmf.end(), 0.));
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != Sum; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, f[deviate] - 1);
         updateSum(&f);
      }
      assert(! f.isValid());
   }
   {
      // Single decaying.
      std::vector<double> pmf(Size, 0.);
      pmf[0] = 1;
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != 100; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, 0.5 * f[deviate]);
         updateSum(&f);
      }
      assert(f.isValid());
   }
   {
      // Single growing.
      std::vector<double> pmf(Size, 0.);
      pmf[0] = 1;
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != 100; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, 2 * f[deviate]);
         updateSum(&f);
      }
      assert(f.isValid());
   }
   {
      // Increasing.
      std::vector<double> pmf(Size);
      for (std::size_t i = 0; i != pmf.size(); ++i) {
         pmf[i] = i;
      }
      const std::size_t Sum =
         static_cast<std::size_t>(std::accumulate(pmf.begin(), pmf.end(), 0.));
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != Sum; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, f[deviate] - 1);
         updateSum(&f);
      }
      assert(! f.isValid());
   }
   {
      // Decreasing.
      std::vector<double> pmf(Size);
      for (std::size_t i = 0; i != pmf.size(); ++i) {
         pmf[i] = Size - i - 1;
      }
      const std::size_t Sum =
         static_cast<std::size_t>(std::accumulate(pmf.begin(), pmf.end(), 0.));
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != Sum; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, f[deviate] - 1);
         updateSum(&f);
      }
      assert(! f.isValid());
   }
} {
   const std::size_t Size = 10;
#ifdef USE_INFLUENCE
   container::StaticArrayOfArrays<std::size_t> influence;
   {
      // No influence.
      std::vector<std::size_t> sizes(Size), values;
      std::fill(sizes.begin(), sizes.end(), 0);
      influence.rebuild(sizes.begin(), sizes.end(), values.begin(), values.end());
   }
#endif

   Generator::DiscreteUniformGenerator uniform;
   Generator f(&uniform);
#ifdef USE_INFLUENCE
   f.setInfluence(&influence);
#endif
   {
      // Constant.
      std::vector<double> pmf(Size, 10.);
      const std::size_t Sum =
         static_cast<std::size_t>(std::accumulate(pmf.begin(), pmf.end(), 0.));
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != Sum; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, f[deviate] - 1);
         updateSum(&f);
      }
      assert(! f.isValid());
   }
   {
      // Increasing.
      std::vector<double> pmf(Size);
      for (std::size_t i = 0; i != pmf.size(); ++i) {
         pmf[i] = i;
      }
      const std::size_t Sum =
         static_cast<std::size_t>(std::accumulate(pmf.begin(), pmf.end(), 0.));
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != Sum; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, f[deviate] - 1);
         updateSum(&f);
      }
      assert(! f.isValid());
   }
   {
      // Decreasing.
      std::vector<double> pmf(Size);
      for (std::size_t i = 0; i != pmf.size(); ++i) {
         pmf[i] = Size - i - 1;
      }
      const std::size_t Sum =
         static_cast<std::size_t>(std::accumulate(pmf.begin(), pmf.end(), 0.));
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != Sum; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, f[deviate] - 1);
         updateSum(&f);
      }
      assert(! f.isValid());
   }
} {
   const std::size_t Size = 10;
#ifdef USE_INFLUENCE
   container::StaticArrayOfArrays<std::size_t> influence;
   {
      // Influence self and next.
      std::vector<std::size_t> sizes(Size), values(2 * Size);
      std::fill(sizes.begin(), sizes.end(), 2);
      for (std::size_t i = 0; i != Size; ++i) {
         values[2*i] = i;
         values[2*i+1] = (i + 1) % Size;
      }
      influence.rebuild(sizes.begin(), sizes.end(), values.begin(), values.end());
   }
#endif

   Generator::DiscreteUniformGenerator uniform;
   Generator f(&uniform);
#ifdef USE_INFLUENCE
   f.setInfluence(&influence);
#endif
   {
      // Constant.
      std::vector<double> pmf(Size, 10.);
      const std::size_t Sum =
         static_cast<std::size_t>(std::accumulate(pmf.begin(), pmf.end(), 0.));
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != Sum; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, f[deviate] - 1);
         updateSum(&f);
      }
      assert(! f.isValid());
   }
   {
      // Increasing.
      std::vector<double> pmf(Size);
      for (std::size_t i = 0; i != pmf.size(); ++i) {
         pmf[i] = i;
      }
      const std::size_t Sum =
         static_cast<std::size_t>(std::accumulate(pmf.begin(), pmf.end(), 0.));
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != Sum; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, f[deviate] - 1);
         updateSum(&f);
      }
      assert(! f.isValid());
   }
   {
      // Decreasing.
      std::vector<double> pmf(Size);
      for (std::size_t i = 0; i != pmf.size(); ++i) {
         pmf[i] = Size - i - 1;
      }
      const std::size_t Sum =
         static_cast<std::size_t>(std::accumulate(pmf.begin(), pmf.end(), 0.));
      f.initialize(pmf.begin(), pmf.end());
      for (std::size_t i = 0; i != Sum; ++i) {
         assert(f.isValid());
         std::size_t deviate = f();
         assert(f[deviate] > 0);
         f.set(deviate, f[deviate] - 1);
         updateSum(&f);
      }
      assert(! f.isValid());
   }
}
