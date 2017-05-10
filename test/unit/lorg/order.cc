// -*- C++ -*-

#include "stlib/lorg/order.h"

using namespace stlib;

void
randomOrder()
{
  const std::size_t Size = 10;
  std::vector<std::size_t> ranking;
  lorg::randomOrder(Size, &ranking);
  assert(ranking.size() == Size);
  std::sort(ranking.begin(), ranking.end());
  for (std::size_t i = 0; i != ranking.size(); ++i) {
    assert(ranking[i] == i);
  }

  std::vector<std::size_t> mapping;
  lorg::randomOrder(Size, &ranking, &mapping);
  assert(ranking.size() == Size);
  assert(mapping.size() == Size);
  for (std::size_t i = 0; i != ranking.size(); ++i) {
    assert(mapping[ranking[i]] == i);
  }
}

template<typename _Float>
void
axis1()
{
  const std::size_t Dimension = 1;
  typedef std::array<_Float, Dimension> Point;

  std::vector<Point> positions;
  std::vector<std::size_t> order, mapping;
  positions.push_back(Point{{0}});
  lorg::axisOrder(positions, &order);
  assert(order[0] == 0);
  lorg::axisOrder(positions, &order, &mapping);
  assert(order[0] == 0);
  assert(mapping[0] == 0);

  positions.clear();
  positions.push_back(Point{{0}});
  positions.push_back(Point{{0.25}});
  positions.push_back(Point{{0.5}});
  positions.push_back(Point{{0.75}});
  lorg::axisOrder(positions, &order, &mapping);
  for (std::size_t i = 0; i != order.size(); ++i) {
    assert(order[i] == i);
  }
  for (std::size_t i = 0; i != mapping.size(); ++i) {
    assert(mapping[i] == i);
  }

  positions.clear();
  positions.push_back(Point{{0.75}});
  positions.push_back(Point{{0.5}});
  positions.push_back(Point{{0.25}});
  positions.push_back(Point{{0}});
  lorg::axisOrder(positions, &order, &mapping);
  for (std::size_t i = 0; i != order.size(); ++i) {
    assert(order[i] == order.size() - i - 1);
  }
  for (std::size_t i = 0; i != mapping.size(); ++i) {
    assert(mapping[i] == mapping.size() - i - 1);
  }
  positions.clear();
  positions.push_back(Point{{0.75}});
  positions.push_back(Point{{0}});
  positions.push_back(Point{{0.25}});
  positions.push_back(Point{{0.5}});
  lorg::axisOrder(positions, &order, &mapping);
  for (std::size_t i = 0; i != order.size(); ++i) {
    assert(order[i] == (i + 1) % order.size());
  }
  for (std::size_t i = 0; i != mapping.size(); ++i) {
    assert(mapping[i] == (i + mapping.size() - 1) % mapping.size());
  }
}

template<typename _Float>
void
axis2()
{
  const std::size_t Dimension = 2;
  typedef std::array<_Float, Dimension> Point;

  std::vector<Point> positions;
  std::vector<std::size_t> order, mapping;
  positions.push_back(Point{{0, 0}});
  lorg::axisOrder(positions, &order);
  assert(order[0] == 0);

  positions.clear();
  positions.push_back(Point{{0, 0}});
  positions.push_back(Point{{0, 0.25}});
  positions.push_back(Point{{0, 0.5}});
  positions.push_back(Point{{0, 0.75}});
  lorg::axisOrder(positions, &order, &mapping);
  for (std::size_t i = 0; i != order.size(); ++i) {
    assert(order[i] == i);
  }
  for (std::size_t i = 0; i != mapping.size(); ++i) {
    assert(mapping[i] == i);
  }

  positions.clear();
  positions.push_back(Point{{0, 0.75}});
  positions.push_back(Point{{0, 0.5}});
  positions.push_back(Point{{0, 0.25}});
  positions.push_back(Point{{0, 0}});
  lorg::axisOrder(positions, &order, &mapping);
  for (std::size_t i = 0; i != order.size(); ++i) {
    assert(order[i] == order.size() - i - 1);
  }
  for (std::size_t i = 0; i != mapping.size(); ++i) {
    assert(mapping[i] == mapping.size() - i - 1);
  }
}

template<typename _Integer, typename _Float>
void
morton1()
{
  const std::size_t Dimension = 1;
  typedef std::array<_Float, Dimension> Point;

  std::vector<Point> positions;
  std::vector<std::size_t> order, mapping;
  positions.push_back(Point{{0}});
  lorg::mortonOrder<_Integer>(positions, &order);
  assert(order[0] == 0);
  lorg::mortonOrder<_Integer>(positions, &order, &mapping);
  assert(order[0] == 0);
  assert(mapping[0] == 0);

  positions.clear();
  positions.push_back(Point{{0}});
  positions.push_back(Point{{0.25}});
  positions.push_back(Point{{0.5}});
  positions.push_back(Point{{0.75}});
  lorg::mortonOrder<_Integer>(positions, &order, &mapping);
  for (std::size_t i = 0; i != order.size(); ++i) {
    assert(order[i] == i);
  }
  for (std::size_t i = 0; i != mapping.size(); ++i) {
    assert(mapping[i] == i);
  }

  positions.clear();
  positions.push_back(Point{{0.75}});
  positions.push_back(Point{{0.5}});
  positions.push_back(Point{{0.25}});
  positions.push_back(Point{{0}});
  lorg::mortonOrder<_Integer>(positions, &order, &mapping);
  for (std::size_t i = 0; i != order.size(); ++i) {
    assert(order[i] == order.size() - i - 1);
  }
  for (std::size_t i = 0; i != mapping.size(); ++i) {
    assert(mapping[i] == mapping.size() - i - 1);
  }
  positions.clear();
  positions.push_back(Point{{0.75}});
  positions.push_back(Point{{0}});
  positions.push_back(Point{{0.25}});
  positions.push_back(Point{{0.5}});
  lorg::mortonOrder<_Integer>(positions, &order, &mapping);
  for (std::size_t i = 0; i != order.size(); ++i) {
    assert(order[i] == (i + 1) % order.size());
  }
  for (std::size_t i = 0; i != mapping.size(); ++i) {
    assert(mapping[i] == (i + mapping.size() - 1) % mapping.size());
  }
}

template<typename _Integer, typename _Float>
void
morton2()
{
  const std::size_t Dimension = 2;
  typedef std::array<_Float, Dimension> Point;

  std::vector<Point> positions;
  std::vector<std::size_t> order, mapping;
  positions.push_back(Point{{0, 0}});
  lorg::mortonOrder<_Integer>(positions, &order);
  assert(order[0] == 0);

  positions.clear();
  positions.push_back(Point{{0.25, 0.25}});
  positions.push_back(Point{{0.75, 0.25}});
  positions.push_back(Point{{0.25, 0.75}});
  positions.push_back(Point{{0.75, 0.75}});
  lorg::mortonOrder<_Integer>(positions, &order, &mapping);
  for (std::size_t i = 0; i != order.size(); ++i) {
    assert(order[i] == i);
  }
  for (std::size_t i = 0; i != mapping.size(); ++i) {
    assert(mapping[i] == i);
  }

  positions.clear();
  positions.push_back(Point{{0.75, 0.75}});
  positions.push_back(Point{{0.25, 0.75}});
  positions.push_back(Point{{0.75, 0.25}});
  positions.push_back(Point{{0.25, 0.25}});
  lorg::mortonOrder<_Integer>(positions, &order, &mapping);
  for (std::size_t i = 0; i != order.size(); ++i) {
    assert(order[i] == order.size() - i - 1);
  }
  for (std::size_t i = 0; i != mapping.size(); ++i) {
    assert(mapping[i] == mapping.size() - i - 1);
  }
}

int
main()
{
  randomOrder();

  axis1<float>();
  axis1<double>();
  axis2<float>();
  axis2<double>();

  morton1<unsigned char, float>();
  morton1<unsigned char, double>();
  morton2<unsigned char, float>();
  morton2<unsigned char, double>();

  morton1<unsigned short, float>();
  morton1<unsigned short, double>();
  morton2<unsigned short, float>();
  morton2<unsigned short, double>();

  morton1<unsigned, float>();
  morton1<unsigned, double>();
  morton2<unsigned, float>();
  morton2<unsigned, double>();

  morton1<std::size_t, float>();
  morton1<std::size_t, double>();
  morton2<std::size_t, float>();
  morton2<std::size_t, double>();

  return 0;
}
