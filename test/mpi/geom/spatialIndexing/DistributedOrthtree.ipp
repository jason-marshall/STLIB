// -*- C++ -*-

#ifndef __test_geom_spatialIndexing_DistributedOrthtree_ipp__
#error This is an implementation detail.
#endif

{
   std::cout << "----------------------------------------------------------\n"
   << "Dimension = " << Dimension
   << ", MaximumLevel = " << MaximumLevel << "\n";
   typedef geom::OrthtreeMap<Dimension, MaximumLevel> Orthtree;
   typedef geom::DistributedOrthtree<Orthtree> DistributedOrthtree;
   typedef DistributedOrthtree::Key Key;
   typedef DistributedOrthtree::Element Element;
   typedef DistributedOrthtree::iterator iterator;
   typedef DistributedOrthtree::Point Point;
   static_assert(DistributedOrthtree::Dimension == Dimension, "Error.");
   static_assert(DistributedOrthtree::MaximumLevel == MaximumLevel, "Error.");
   static_assert(DistributedOrthtree::NumberOfOrthants == 1 << Dimension,
                 "Error.");
   {
      // Constructor.
      Point lowerCorner = ext::filled_array<Point>(0.),
      extents = ext::filled_array<Point>(1.);
      DistributedOrthtree x(MPI::COMM_WORLD, lowerCorner, extents);
      // Accessors.
      assert(x.begin() == x.end());
      assert(x.size() == 0);
      assert(x.max_size() > 0);
      assert(x.empty());
      assert(x.isBalanced());
      // Search.
      assert(x.count(Key()) == 0);
      // Insert.
      iterator i = x.insert(Key());
      assert(i->first == Key());
      assert(i->second == Element());
      assert(x.begin()->first == Key());
      assert(x.begin()->second == Element());
      assert(x.begin() != x.end());
      assert(x.size() == 1);
      assert(! x.empty());
      // Search.
      assert(x.count(Key()) == 1);
      assert(x.find(Key()) == i);
      // Erase through iterator.
      x.erase(i);
      assert(x.empty());
      // Erase through key.
      x.insert(Key());
      assert(x.erase(Key()) == 1);
      assert(x.empty());
   }
}

// End of file.
