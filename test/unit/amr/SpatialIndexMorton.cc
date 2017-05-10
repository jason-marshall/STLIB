// -*- C++ -*-

#include "stlib/amr/SpatialIndexMorton.h"

#include <vector>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

template<std::size_t Dimension, std::size_t MaximumLevel>
inline
void
test()
{
  typedef amr::SpatialIndexMorton<Dimension, MaximumLevel> Index;
  typedef typename Index::Level Level;
  typedef typename Index::Coordinate Coordinate;
  typedef typename Index::Code Code;
  typedef typename Index::CoordinateList CoordinateList;
  static_assert(Index::Dimension == Dimension, "Bad dimension.");
  static_assert(Index::MaximumLevel == MaximumLevel, "Bad maximum level.");
  static_assert(Index::NumberOfOrthants == 1 << Dimension,
                "Bad number of orthants.");
  static_assert(1 << std::numeric_limits<Level>::digits >= MaximumLevel,
                "Bad level.");
  static_assert(std::numeric_limits<Coordinate>::digits >=
                int(MaximumLevel), "Bad coordinate.");
  static_assert(std::numeric_limits<Code>::digits >=
                int(Dimension * MaximumLevel + 1), "Bad code.");
  std::cout << "----------------------------------------------------------\n"
            << "Dimension = " << Dimension
            << ", MaximumLevel = " << MaximumLevel << "\n"
            << "Digits in Level = "
            << std::numeric_limits<Level>::digits << "\n"
            << "Digits in Coordinate = "
            << std::numeric_limits<Coordinate>::digits << "\n"
            << "Digits in Code = "
            << std::numeric_limits<Code>::digits << "\n";
  {
    // Default constructor.
    Index x;
    assert(x.getLevel() == 0);
    for (std::size_t i = 0; i != Dimension; ++i) {
      assert(x.getCoordinates()[i] == 0);
    }
  }
  {
    // Validity.
    Index x;
    assert(x.isValid());
    x.invalidate();
    assert(! x.isValid());
    //std::cout << "Invalid =\n" << x << "\n";
    // The invalid index is greater than all others.
    Index y;
    assert(y < x);
    CoordinateList coordinates;
    std::fill(coordinates.begin(), coordinates.end(), -1);
    Index z(MaximumLevel, coordinates);
    assert(z < x);
    //std::cout << "Last =\n" << z << "\n";
  }
  {
    // Construct from level and coordinates.
    CoordinateList coordinates;
    std::fill(coordinates.begin(), coordinates.end(), 0);
    for (std::size_t i = 0; i != Dimension; ++i) {
      coordinates[i] = i;
    }
    Index x(MaximumLevel, coordinates);
    assert(x.getLevel() == MaximumLevel);
    assert(x.getCoordinates() == coordinates);
    {
      // Copy constructor.
      Index y(x);
      assert(x == y);
    }
    {
      // Assignment operator.
      Index y;
      y = x;
      assert(x == y);
    }
    {
      // Manipulators.
      Index y;
      y.set(MaximumLevel, coordinates);
      assert(y.getLevel() == MaximumLevel);
      assert(x.getCoordinates() == coordinates);
      assert(x == y);
    }
    // Print.
    std::cout << x << "\n\n";
  }

  // Parent and children.
  {
    Index parent;
    Index child;

    assert(parent.getLevel() == 0);
    for (std::size_t i = 0; i != Dimension; ++i) {
      assert(parent.getCoordinates()[i] == 0);
    }
    std::cout << parent << "\n";

    child = parent;
    for (std::size_t i = 0; i != Index::NumberOfOrthants; ++i) {
      child.transformToChild(i);
      std::cout << child << "\n";
      assert(child.getLevel() == 1);
      for (std::size_t d = 0; d != Dimension; ++d) {
        assert(child.getCoordinates()[d] == (i >> d) % 2);
      }
      child.transformToParent();
      assert(parent == child);
    }

    for (std::size_t i = 0; i != Index::NumberOfOrthants; ++i) {
      child.transformToChild(i);
      for (std::size_t j = 0; j != Index::NumberOfOrthants; ++j) {
        child.transformToChild(j);
        std::cout << child << "\n";
        assert(child.getLevel() == 2);
        for (std::size_t d = 0; d != Dimension; ++d) {
          assert(child.getCoordinates()[d] == 2 * ((i >> d) % 2) +
                 (j >> d) % 2);
        }
        child.transformToParent();
      }
      child.transformToParent();
      assert(parent == child);
    }
  }
  // Comparison.
  {
    Index x;
    x.transformToChild(0);
    assert(x == x);
    assert(x.getCode() == x);
    assert(x == x.getCode());
    Index y;
    y.transformToChild(1);
    assert(x != y);
    assert(x.getCode() != y);
    assert(x != y.getCode());

    assert(x < y);
    assert(x.getCode() < y);
    assert(x < y.getCode());

    assert(x <= y);
    assert(x.getCode() <= y);
    assert(x <= y.getCode());

    assert(y > x);
    assert(y.getCode() > x);
    assert(y > x.getCode());

    assert(y >= x);
    assert(y.getCode() >= x);
    assert(y >= x.getCode());
  }
  // Neighbor
  {
    Index node, neighbor;
    for (std::size_t i = 0; i != 2 * Dimension; ++i) {
      assert(! hasNeighbor(node, i));
    }
    node.transformToChild(0);
    node.transformToChild(0);
    for (std::size_t d = 0; d != Dimension; ++d) {
      assert(! hasNeighbor(node, 2 * d));
      assert(hasNeighbor(node, 2 * d + 1));
      neighbor = node;
      neighbor.transformToNeighbor(2 * d + 1);
      assert(node != neighbor);
      assert(amr::areAdjacent(node, neighbor));
      neighbor.transformToNeighbor(2 * d);
      assert(node == neighbor);
      assert(! amr::areAdjacent(node, neighbor));

      neighbor.transformToNeighbor(2 * d + 1);
      neighbor.transformToChild(0);
      assert(amr::areAdjacent(node, neighbor));
      neighbor.transformToChild(0);
      assert(amr::areAdjacent(node, neighbor));
      neighbor.transformToParent();
      neighbor.transformToParent();
      neighbor.transformToNeighbor(2 * d);
      assert(node == neighbor);

      neighbor.transformToNeighbor(2 * d + 1);
      neighbor.transformToNeighbor(2 * d + 1);
      assert(! amr::areAdjacent(node, neighbor));
      neighbor.transformToNeighbor(2 * d);
      neighbor.transformToNeighbor(2 * d);
      assert(node == neighbor);

      std::size_t e = (d + 1) % Dimension;
      neighbor.transformToNeighbor(2 * d + 1);
      neighbor.transformToNeighbor(2 * e + 1);
      assert(! amr::areAdjacent(node, neighbor));
      neighbor.transformToNeighbor(2 * e);
      neighbor.transformToNeighbor(2 * d);
      assert(node == neighbor);
    }
  }
  // Location.
  {
    Index x;
    CoordinateList location;
    amr::computeLocation(x, &location);
    CoordinateList position;
    std::fill(position.begin(), position.end(), 0);
    assert(location == position);
    x.transformToChild(Index::NumberOfOrthants - 1);
    amr::computeLocation(x, &location);
    position += Coordinate(1 << (MaximumLevel - 1));
    assert(location == position);
    x.transformToChild(Index::NumberOfOrthants - 1);
    amr::computeLocation(x, &location);
    position += Coordinate(1 << (MaximumLevel - 2));
    assert(location == position);
  }
  // Length.
  {
    Index x;
    std::size_t length = 1 << MaximumLevel;
    assert(amr::computeLength(x) == length);
    x.transformToChild(Index::NumberOfOrthants - 1);
    length /= 2;
    assert(amr::computeLength(x) == length);
    x.transformToChild(Index::NumberOfOrthants - 1);
    length /= 2;
    assert(amr::computeLength(x) == length);
    while (x.canBeRefined()) {
      x.transformToChild(Index::NumberOfOrthants - 1);
    }
    assert(amr::computeLength(x) == 1);
  }
  // Separations.
  {
    Index a, b;
    int length = 1 << MaximumLevel;
    int pointDistance = 0;
    std::array<int, Dimension> separations;
    amr::computeSeparations(a, b, &separations);
    {
      std::array<int, Dimension> x;
      std::fill(x.begin(), x.end(), pointDistance - length);
      assert(separations == x);
    }

    a.transformToChild(0);
    b.transformToChild(Index::NumberOfOrthants - 1);
    length /= 2;
    pointDistance = length;
    amr::computeSeparations(a, b, &separations);
    for (std::size_t d = 0; d != Dimension; ++d) {
      assert(separations[d] == pointDistance - length);
    }

    while (length != 1) {
      a.transformToChild(0);
      b.transformToChild(0);
      length /= 2;
      amr::computeSeparations(a, b, &separations);
      for (std::size_t d = 0; d != Dimension; ++d) {
        assert(separations[d] == pointDistance - length);
      }
      amr::computeSeparations(b, a, &separations);
      for (std::size_t d = 0; d != Dimension; ++d) {
        assert(separations[d] == pointDistance - length);
      }
    }
  }
  // hasNext.
  {
    Index x;
    assert(! amr::hasNext(x));

    for (std::size_t i = 0; i != Index::NumberOfOrthants; ++i) {
      x.transformToChild(i);
      if (i == Index::NumberOfOrthants - 1) {
        assert(! amr::hasNext(x));
      }
      else {
        assert(amr::hasNext(x));
      }
      x.transformToParent();
    }

    for (std::size_t i = 0; i != Index::NumberOfOrthants; ++i) {
      x.transformToChild(i);
      for (std::size_t j = 0; j != Index::NumberOfOrthants; ++j) {
        x.transformToChild(j);
        if (i == Index::NumberOfOrthants - 1 &&
            j == Index::NumberOfOrthants - 1) {
          assert(! amr::hasNext(x));
        }
        else {
          assert(amr::hasNext(x));
        }
        x.transformToParent();
      }
      x.transformToParent();
    }
  }
  // transformToNext.
  {
    Index x, y;

    // Do loops to cycle through all nodes on a level.

    // Level 0.
    y.transformToNext();
    assert(y == x);

    // Level 1.
    x.transformToChild(0);
    y = x;
    for (std::size_t i = 0; i != Index::NumberOfOrthants; ++i) {
      y.transformToNext();
    }
    assert(y == x);

    // Level 2.
    x.transformToChild(0);
    y = x;
    for (std::size_t i = 0; i != Index::NumberOfOrthants *
         Index::NumberOfOrthants; ++i) {
      y.transformToNext();
    }
    assert(y == x);
  }
  // Message stream I/O.
  {
    Index x;
    x.transformToChild(1);
    amr::MessageOutputStream out(x.getMessageStreamSize());
    out << x;
    amr::MessageInputStream in(out);
    Index y;
    in >> y;
    assert(x == y);

    // Invalid index.
    x.invalidate();
    out.clear();
    out << x;
    in = out;
    in >> y;
    assert(x == y);
  }
  {
    Index x;
    x.transformToChild(1);
    amr::MessageOutputStreamChecked out;
    out << x;
    amr::MessageInputStream in(out);
    Index y;
    in >> y;
    assert(x == y);

    // Invalid index.
    x.invalidate();
    out.clear();
    out << x;
    in = out;
    in >> y;
    assert(x == y);
  }
  //-------------------------------------------------------------------------
  // Topology and geometry.
  {
    // hasParent, hasChildren
    Index x;
    assert(! hasParent(x));
    assert(hasChildren(x));
    while (x.getLevel() < MaximumLevel - 1) {
      x.transformToChild(0);
      assert(hasParent(x));
      assert(hasChildren(x));
    }
    x.transformToChild(0);
    assert(x.getLevel() == MaximumLevel);
    assert(hasParent(x));
    assert(! hasChildren(x));
  }
  {
    // getAdjacentNeighbors
    Index x;
    std::vector<Index> neighbors;
    getAdjacentNeighbors(x, std::back_inserter(neighbors));
    assert(neighbors.empty());

    x.transformToChild(0);
    getAdjacentNeighbors(x, std::back_inserter(neighbors));
    assert(neighbors.size() == Dimension);
    neighbors.clear();

    x.transformToChild(Index::NumberOfOrthants - 1);
    getAdjacentNeighbors(x, std::back_inserter(neighbors));
    assert(neighbors.size() == 2 * Dimension);
    neighbors.clear();
  }
  {
    // getAdjacentNeighborsHigherLevel
    Index x;
    std::vector<Index> neighbors;
    getAdjacentNeighborsHigherLevel(x, std::back_inserter(neighbors));
    assert(neighbors.empty());

    x.transformToChild(0);
    getAdjacentNeighborsHigherLevel(x, std::back_inserter(neighbors));
    assert(neighbors.size() == Dimension * Index::NumberOfOrthants / 2);
    for (std::size_t i = 0; i != neighbors.size(); ++i) {
      assert(areAdjacent(x, neighbors[i]));
    }
    neighbors.clear();

    x.transformToChild(Index::NumberOfOrthants - 1);
    getAdjacentNeighborsHigherLevel(x, std::back_inserter(neighbors));
    assert(neighbors.size() == Dimension * Index::NumberOfOrthants);
    for (std::size_t i = 0; i != neighbors.size(); ++i) {
      assert(areAdjacent(x, neighbors[i]));
    }
    neighbors.clear();
  }
}

int
main()
{
  test<1, 10>();
  test<2, 8>();
  test<3, 6>();
  test<4, 4>();

  return 0;
}
