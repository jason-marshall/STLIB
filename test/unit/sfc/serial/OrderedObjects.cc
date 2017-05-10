// -*- C++ -*-

#include "stlib/sfc/OrderedObjects.h"


int
main()
{
  typedef std::uint64_t Code;
  typedef std::vector<std::pair<Code, std::size_t> > VectorPair;
  using stlib::sfc::OrderedObjects;

  {
    OrderedObjects orderedObjects;
    orderedObjects.set(VectorPair{});
    {
      std::vector<std::size_t> indices;
      orderedObjects.mapToOriginalIndices(indices.begin(), indices.end());
      assert(indices.empty());
    }
  }

  {
    OrderedObjects orderedObjects;
    orderedObjects.set(VectorPair{{0, 1}, {0, 2}, {0, 0}});
    {
      std::vector<std::size_t> indices;
      orderedObjects.mapToOriginalIndices(indices.begin(), indices.end());
      assert(indices.empty());
    }
    {
      std::vector<std::size_t> indices = {0, 1, 2};
      orderedObjects.mapToOriginalIndices(indices.begin(), indices.end());
      assert(indices == (std::vector<std::size_t>{1, 2, 0}));
    }
    {
      std::vector<std::size_t> indices = {2, 1, 0};
      orderedObjects.mapToOriginalIndices(indices.begin(), indices.end());
      assert(indices == (std::vector<std::size_t>{0, 2, 1}));
    }
    {
      std::vector<std::size_t> indices = {0, 0, 0};
      orderedObjects.mapToOriginalIndices(indices.begin(), indices.end());
      assert(indices == (std::vector<std::size_t>{1, 1, 1}));
    }
  }

  return 0;
}
