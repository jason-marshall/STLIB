// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueuePlaceboQueue.h"

using namespace stlib;

int
main()
{
  typedef ads::IndexedPriorityQueuePlaceboQueue<> IndexedPriorityQueue;
  typedef IndexedPriorityQueue::Key Key;

  Key keys[] = {9, 8, 7, 6, 5 , 4, 3, 2, 1, 0};
  const int size = sizeof(keys) / sizeof(Key);
  std::vector<Key> propensities(size, Key(1));
  IndexedPriorityQueue x(size);

  // Push.
  //std::cerr << "Push.\n";
  for (int i = 0; i != size; ++i) {
    x.push(i, keys[i]);
  }

  // Get.
  //std::cerr << "Get.\n";
  for (int i = 0; i != size; ++i) {
    assert(x.get(i) == keys[i]);
  }
  x.clear();

  // PushTop.
  //std::cerr << "PushTop.\n";
  for (int i = 0; i != size; ++i) {
    x.push(i, keys[i]);
  }
  for (int i = 0; i != size; ++i) {
    assert(x.top() < size);
    x.pushTop(0.1 * i);
  }
  x.clear();

  // Set.
  //std::cerr << "Set.\n";
  for (int i = 0; i != size; ++i) {
    x.push(i, keys[i]);
  }
  for (int i = 0; i != size; ++i) {
    keys[i] *= 2;
  }
  for (int i = 0; i != size; ++i) {
    x.set(i, keys[i]);
  }
  for (int i = 0; i != size; ++i) {
    assert(x.get(i) == keys[i]);
  }
  x.clear();

  return 0;
}
