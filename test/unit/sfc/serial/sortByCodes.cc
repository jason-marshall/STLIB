// -*- C++ -*-

#include "stlib/sfc/sortByCodes.h"


template<typename _Code, typename _Object>
void
test(std::vector<_Object> const& objects)
{
  stlib::sfc::OrderedObjects orderedObjects;
  std::vector<_Object> copy = objects;
  stlib::sfc::sortByCodes<_Code>(&copy, &orderedObjects);
  orderedObjects.restore(copy.begin(), copy.end());
  assert(copy == objects);
}


template<typename _Object>
void
testEach(std::vector<_Object> const& objects)
{
  test<std::uint8_t>(objects);
  test<std::uint16_t>(objects);
  test<std::uint32_t>(objects);
  test<std::uint64_t>(objects);
}


int
main()
{
  {
    typedef std::array<float, 3> Object;
    testEach(std::vector<Object>{});
    testEach(std::vector<Object>{{{0, 0, 0}}});
    testEach(std::vector<Object>{{{1, 1, 1}}, {{0, 0, 0}}});
    testEach(std::vector<Object>{{{1, 1, 1}}, {{0, 0, 0}}, {{1, 2, 3}},
                                 {{2, 3, 1}}, {{3, 1, 2}}, {{2, 1, 3}}});
  }

  return 0;
}
