// -*- C++ -*-

#include "stlib/particle/orderMpi.h"
#include "stlib/ads/functor/Identity.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"


using namespace stlib;

template<typename _Traits>
class MortonOrder :
  public particle::MortonOrder<_Traits>
{
public:
  typedef typename _Traits::Float Float;
  typedef particle::MortonOrder<_Traits> Base;

  MortonOrder(const geom::BBox<Float, _Traits::Dimension>& domain,
              const Float interactionDistance, const Float padding) :
    Base(domain, interactionDistance, padding)
  {
  }

  using Base::morton;
  using Base::codes;
};


void
testMerge()
{
  typedef particle::IntegerTypes::Code Code;

  std::vector<std::pair<Code, std::size_t> > a, b, merged;

  particle::merge(a, b, &merged);
  assert(merged.empty());

  // (0, 1)
  //
  // (0, 1)
  a.push_back(std::make_pair(Code(0), std::size_t(1)));
  particle::merge(a, b, &merged);
  assert(merged.size() == 1);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  particle::merge(b, a, &merged);
  assert(merged.size() == 1);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);

  // (0, 1)
  // (2, 2)
  // (0, 1), (2, 2)
  b.push_back(std::make_pair(Code(2), std::size_t(2)));
  particle::merge(a, b, &merged);
  assert(merged.size() == 2);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  assert(merged[1].first == 2);
  assert(merged[1].second == 2);
  particle::merge(b, a, &merged);
  assert(merged.size() == 2);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  assert(merged[1].first == 2);
  assert(merged[1].second == 2);

  // (0, 1), (1, 2)
  // (2, 2)
  // (0, 1), (1, 2), (2, 2)
  a.push_back(std::make_pair(Code(1), std::size_t(2)));
  particle::merge(a, b, &merged);
  assert(merged.size() == 3);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  assert(merged[1].first == 1);
  assert(merged[1].second == 2);
  assert(merged[2].first == 2);
  assert(merged[2].second == 2);
  particle::merge(b, a, &merged);
  assert(merged.size() == 3);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  assert(merged[1].first == 1);
  assert(merged[1].second == 2);
  assert(merged[2].first == 2);
  assert(merged[2].second == 2);

  // (0, 1), (1, 2)
  // (2, 2), (3, 3)
  // (0, 1), (1, 2), (2, 2), (3, 3)
  b.push_back(std::make_pair(Code(3), std::size_t(3)));
  particle::merge(a, b, &merged);
  assert(merged.size() == 4);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  assert(merged[1].first == 1);
  assert(merged[1].second == 2);
  assert(merged[2].first == 2);
  assert(merged[2].second == 2);
  assert(merged[3].first == 3);
  assert(merged[3].second == 3);
  particle::merge(b, a, &merged);
  assert(merged.size() == 4);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  assert(merged[1].first == 1);
  assert(merged[1].second == 2);
  assert(merged[2].first == 2);
  assert(merged[2].second == 2);
  assert(merged[3].first == 3);
  assert(merged[3].second == 3);

  // (0, 1), (1, 2), (2, 3)
  // (2, 2), (3, 3)
  // (0, 1), (1, 2), (2, 5), (3, 3)
  a.push_back(std::make_pair(Code(2), std::size_t(3)));
  particle::merge(a, b, &merged);
  assert(merged.size() == 4);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  assert(merged[1].first == 1);
  assert(merged[1].second == 2);
  assert(merged[2].first == 2);
  assert(merged[2].second == 5);
  assert(merged[3].first == 3);
  assert(merged[3].second == 3);
  particle::merge(b, a, &merged);
  assert(merged.size() == 4);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  assert(merged[1].first == 1);
  assert(merged[1].second == 2);
  assert(merged[2].first == 2);
  assert(merged[2].second == 5);
  assert(merged[3].first == 3);
  assert(merged[3].second == 3);

  // (0, 1), (1, 2), (2, 3)
  // (2, 2), (3, 3), (5, 5)
  // (0, 1), (1, 2), (2, 5), (3, 3), (5, 5)
  b.push_back(std::make_pair(Code(5), std::size_t(5)));
  particle::merge(a, b, &merged);
  assert(merged.size() == 5);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  assert(merged[1].first == 1);
  assert(merged[1].second == 2);
  assert(merged[2].first == 2);
  assert(merged[2].second == 5);
  assert(merged[3].first == 3);
  assert(merged[3].second == 3);
  assert(merged[4].first == 5);
  assert(merged[4].second == 5);
  particle::merge(b, a, &merged);
  assert(merged.size() == 5);
  assert(merged[0].first == 0);
  assert(merged[0].second == 1);
  assert(merged[1].first == 1);
  assert(merged[1].second == 2);
  assert(merged[2].first == 2);
  assert(merged[2].second == 5);
  assert(merged[3].first == 3);
  assert(merged[3].second == 3);
  assert(merged[4].first == 5);
  assert(merged[4].second == 5);
}


template<std::size_t _Dimension>
void
testShift()
{
  typedef particle::IntegerTypes::Code Code;

  std::vector<std::pair<Code, std::size_t> > table;
  const std::size_t MaxShift = 2;
  for (std::size_t shift = 1; shift <= MaxShift; ++shift) {
    const Code Size = 1 << _Dimension * MaxShift;
    table.clear();
    for (Code i = 0; i != Size; ++i) {
      table.push_back(std::make_pair(i, std::size_t(1)));
    }
    particle::shift<_Dimension>(&table, shift);
    assert(table.size() == Size / (1 << _Dimension * shift));
    for (std::size_t i = 0; i != table.size(); ++i) {
      assert(table[i].first == i);
      assert(table[i].second == std::size_t(1) << _Dimension * shift);
    }
  }
}


int
main()
{
  testMerge();
  testShift<1>();
  testShift<2>();
  testShift<3>();

  return 0;
}
