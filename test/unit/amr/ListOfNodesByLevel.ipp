// -*- C++ -*-

#ifndef __test_amr_ListOfNodesByLevel_ipp__
#error This is an implementation detail.
#endif

{
   std::cout << "----------------------------------------------------------\n"
   << "Dimension = " << Dimension
   << ", MaximumLevel = " << MaximumLevel << "\n";
   typedef amr::ListOfNodesByLevel<MaximumLevel, std::size_t,
                                   ads::Identity<std::size_t> >
   ListOfNodesByLevel;
   static_assert(ListOfNodesByLevel::MaximumLevel == MaximumLevel,
                 "Bad maximum level.");
   {
      // Constructor.
      ListOfNodesByLevel x;
      // Accessors.
      assert(x.begin() == x.end());
      for (int level = 0; level <= MaximumLevel; ++level) {
         assert(x.begin(level) == x.begin());
         assert(x.end(level) == x.begin());
      }
      assert(x.size() == 0);
      for (int level = 0; level <= MaximumLevel; ++level) {
         assert(x.size(level) == 0);
      }
      assert(x.max_size() > 0);
      assert(x.empty());
      assert(x.isValid());

      // Insert.
      x.insert(0);
      x.insert(1);
      x.insert(2);
      assert(x.isValid());
      assert(x.size() == 3);
      assert(x.size(0) == 1);
      assert(x.size(1) == 1);
      assert(x.size(2) == 1);
      x.clear();
      assert(x.empty());
      assert(x.isValid());

      x.insert(3);
      x.insert(3);
      x.insert(3);
      x.insert(2);
      x.insert(2);
      x.insert(1);
      assert(x.isValid());
      assert(x.size() == 6);
      assert(x.size(0) == 0);
      assert(x.size(1) == 1);
      assert(x.size(2) == 2);
      assert(x.size(3) == 3);
      x.clear();
      assert(x.empty());
      assert(x.isValid());

      // Erase.
      x.insert(0);
      x.insert(1);
      x.insert(2);
      x.erase(x.begin(0));
      assert(x.isValid());
      assert(x.size(0) == 0);
      assert(x.size(1) == 1);
      assert(x.size(2) == 1);
      x.erase(x.begin(1));
      assert(x.isValid());
      assert(x.size(0) == 0);
      assert(x.size(1) == 0);
      assert(x.size(2) == 1);
      x.erase(x.begin(2));
      assert(x.isValid());
      assert(x.size(0) == 0);
      assert(x.size(1) == 0);
      assert(x.size(2) == 0);

      x.insert(0);
      x.insert(1);
      x.insert(2);
      x.erase(x.begin(2));
      assert(x.isValid());
      assert(x.size(0) == 1);
      assert(x.size(1) == 1);
      assert(x.size(2) == 0);
      x.erase(x.begin(1));
      assert(x.isValid());
      assert(x.size(0) == 1);
      assert(x.size(1) == 0);
      assert(x.size(2) == 0);
      x.erase(x.begin(0));
      assert(x.isValid());
      assert(x.size(0) == 0);
      assert(x.size(1) == 0);
      assert(x.size(2) == 0);
   }
}
