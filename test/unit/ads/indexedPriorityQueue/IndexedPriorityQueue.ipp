// -*- C++ -*-

#ifndef __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
#error This is an implementation detail.
#endif

{
   typedef IndexedPriorityQueue::Key Key;

   {
      const std::size_t size = 10;
#ifdef HASHING
      IndexedPriorityQueue x(size, 4 /*hash table size*/, 1 /*target load*/);
#else
      IndexedPriorityQueue x(size);
#endif
   }

   {
      std::vector<Key> keys(10);
      for (std::size_t i = 0; i != keys.size(); ++i) {
         keys[i] = keys.size() - i - 1;
      }
      std::vector<Key> propensities(keys.size(), Key(1));
#ifdef HASHING
      IndexedPriorityQueue x(keys.size(), 4 /*hash table size*/,
      1 /*target load*/);
#else
      IndexedPriorityQueue x(keys.size());
#endif

#ifdef PROPENSITIES
      x.setPropensities(&propensities);
#endif

      // Push.
      //std::cerr << "Push.\n";
      for (std::size_t i = 0; i != keys.size(); ++i) {
         x.push(i, keys[i]);
      }

      // Get.
      //std::cerr << "Get.\n";
      for (std::size_t i = 0; i != keys.size(); ++i) {
         assert(x.get(i) == keys[i]);
      }

      // Pop top.
      //std::cerr << "Pop top.\n";
      for (std::size_t i = 0; i != keys.size(); ++i) {
         const int n = x.top();
         assert(x.get(n) == i);
         x.popTop();
      }
      std::fill(propensities.begin(), propensities.end(), 0);
      assert(x.get(x.top()) == std::numeric_limits<Key>::max());
      std::fill(propensities.begin(), propensities.end(), 1);
      x.clear();

      // Clear.
      //std::cerr << "Clear.\n";
      for (std::size_t i = 0; i != keys.size(); ++i) {
         x.push(i, keys[i]);
      }
      x.clear();
      std::fill(propensities.begin(), propensities.end(), 0);
      assert(x.get(x.top()) == std::numeric_limits<Key>::max());
      std::fill(propensities.begin(), propensities.end(), 1);

      // Pop.
      //std::cerr << "Pop.\n";
      for (std::size_t i = 0; i != keys.size(); ++i) {
         x.push(i, keys[i]);
      }
      for (std::size_t i = 0; i != keys.size(); ++i) {
         x.pop(i);
      }
      std::fill(propensities.begin(), propensities.end(), 0);
      assert(x.get(x.top()) == std::numeric_limits<Key>::max());
      std::fill(propensities.begin(), propensities.end(), 1);
      x.clear();

      // PushTop.
      //std::cerr << "PushTop.\n";
      for (std::size_t i = 0; i != keys.size(); ++i) {
         x.push(i, keys[i]);
      }
      for (std::size_t i = 0; i != keys.size(); ++i) {
         assert(std::size_t(x.top()) == keys.size() - 1);
         x.pushTop(0.1 * i);
      }
      x.clear();

      // Shift.
      for (std::size_t i = 0; i != keys.size(); ++i) {
         x.push(i, keys[i]);
      }
      x.shift(10.);
      for (std::size_t i = 0; i != keys.size(); ++i) {
         assert(x.get(x.top()) == i + 10.);
         x.popTop();
      }
      x.clear();

      // Set.
      //std::cerr << "Set.\n";
      for (std::size_t i = 0; i != keys.size(); ++i) {
         x.push(i, keys[i]);
      }
      for (std::size_t i = 0; i != keys.size(); ++i) {
         x.set(i, 2 * keys[i]);
      }
      for (std::size_t i = 0; i != keys.size(); ++i) {
         assert(x.get(i) == 2 * keys[i]);
      }
      for (std::size_t i = 0; i != keys.size(); ++i) {
         const int n = x.top();
         assert(x.get(n) == 2 * i);
         x.popTop();
      }
      x.clear();

      for (std::size_t i = 0; i != keys.size(); ++i) {
         x.push(i, keys[i]);
      }
      for (std::size_t i = 0; i != keys.size(); ++i) {
         x.set(i, 0.5 * keys[i]);
      }
      for (std::size_t i = 0; i != keys.size(); ++i) {
         assert(x.get(i) == 0.5 * keys[i]);
      }
      for (std::size_t i = 0; i != keys.size(); ++i) {
         const int n = x.top();
         assert(x.get(n) == 0.5 * i);
         x.popTop();
      }
      x.clear();
      // CONTINUE
      // When using MSVC '13, at least on 32-bit architectures, the program
      // crashes here for IndexedPriorityQueuePartitionLinearSearchPropensities.
      // It must be dying in a destructor, however I don't know where.
   }
}
