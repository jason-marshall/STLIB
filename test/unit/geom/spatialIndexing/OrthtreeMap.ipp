// -*- C++ -*-

#ifndef __test_geom_spatialIndexing_SpatialIndex_ipp__
#error This is an implementation detail.
#endif

{
   std::cout << "----------------------------------------------------------\n"
   << "Dimension = " << Dimension
   << ", MaximumLevel = " << MaximumLevel << "\n";
   typedef geom::OrthtreeMap<Dimension, MaximumLevel> Orthtree;
   typedef Orthtree::Key Key;
   typedef Orthtree::Element Element;
   typedef Orthtree::iterator iterator;
   typedef Orthtree::Point Point;
   static_assert(Orthtree::Dimension == Dimension, "Error.");
   static_assert(Orthtree::MaximumLevel == MaximumLevel, "Error.");
   static_assert(Orthtree::NumberOfOrthants == 1 << Dimension, "Error.");
   {
      // Constructor.
      Point lowerCorner, extents;
      std::fill(lowerCorner.begin(), lowerCorner.end(), 0.);
      std::fill(extents.begin(), extents.end(), 1.);
      Orthtree x(lowerCorner, extents);
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
      // Print.
      std::cout << x;
      geom::printVtkUnstructuredGrid(std::cout, x);
      // Erase through iterator.
      x.erase(i);
      assert(x.empty());
      // Erase through key.
      x.insert(Key());
      assert(x.erase(Key()) == 1);
      assert(x.empty());

      // Refine.
      x.insert(Key());
      while (x.canBeRefined(x.begin())) {
         x.split(x.begin());
      }
      assert(x.isBalanced());
      std::cout << "Size after refinement of the first node = " << x.size()
      << "\n";
      x.balance();
      assert(x.isBalanced());
      std::cout << "Size after balancing = " << x.size() << "\n";

      x.clear();
      x.insert(Key());
      x.split(x.begin());
      while (x.canBeRefined(++x.begin())) {
         x.split(++x.begin());
      }
      assert(! x.isBalanced());
      std::cout << "Size after refinement of the second node = " << x.size()
      << "\n";
      x.balance();
      assert(x.isBalanced());
      std::cout << "Size after balancing = " << x.size() << "\n";
   }
}

// End of file.
