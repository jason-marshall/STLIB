// -*- C++ -*-

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueHashing.h"
#include "stlib/ads/indexedPriorityQueue/HashingChainingVector.h"

#define HASHING
#define PROPENSITIES

using namespace stlib;

int
main()
{
  {
    typedef ads::IndexedPriorityQueueHashing<double, ads::HashingChainingVector<std::vector<double>::const_iterator> >
    IndexedPriorityQueue;
#define __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
#include "IndexedPriorityQueue.ipp"
#undef __test_ads_indexedPriorityQueue_IndexedPriorityQueue_ipp__
  }

  return 0;
}
