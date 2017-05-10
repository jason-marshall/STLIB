// -*- C++ -*-

#ifndef __test_geom_spatialIndex_SpatialIndex_ipp__
#error This file is an implementation detail.
#endif

{
  USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
  
  typedef geom::SpatialIndex<Dimension, MaximumLevel> Index;
  typedef Index::Level Level;
  typedef Index::Coordinate Coordinate;
  typedef Index::Code Code;
  static_assert(Index::Dimension == Dimension, "Error.");
  static_assert(Index::MaximumLevel == MaximumLevel, "Error.");
  static_assert(Index::NumberOfOrthants == 1 << Dimension, "Error.");
  static_assert(1 << std::numeric_limits<Level>::digits >= MaximumLevel,
                "Error.");
  static_assert(std::numeric_limits<Coordinate>::digits >= MaximumLevel,
                "Error.");
  static_assert(std::numeric_limits<Code>::digits >=
                Dimension* MaximumLevel, "Error.");
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
    for (int i = 0; i != Dimension; ++i) {
      assert(x.getCoordinates()[i] == 0);
    }
  }
  {
    // Construct from level and coordinates.
    std::array<Coordinate, Dimension> Coordinates;
    for (int i = 0; i != Dimension; ++i) {
      Coordinates[i] = i;
    }
    Index x(MaximumLevel, Coordinates);
    assert(x.getLevel() == MaximumLevel);
    assert(x.getCoordinates() == Coordinates);
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
      y.set(MaximumLevel, Coordinates);
      assert(y.getLevel() == MaximumLevel);
      assert(x.getCoordinates() == Coordinates);
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
    for (int i = 0; i != Dimension; ++i) {
      assert(parent.getCoordinates()[i] == 0);
    }
    std::cout << parent << "\n";

    child = parent;
    for (int i = 0; i != Index::NumberOfOrthants; ++i) {
      child.transformToChild(i);
      std::cout << child << "\n";
      assert(child.getLevel() == 1);
      for (int d = 0; d != Dimension; ++d) {
        assert(child.getCoordinates()[d] == (i >> d) % 2);
      }
      child.transformToParent();
      assert(parent == child);
    }

    for (int i = 0; i != Index::NumberOfOrthants; ++i) {
      child.transformToChild(i);
      for (int j = 0; j != Index::NumberOfOrthants; ++j) {
        child.transformToChild(j);
        std::cout << child << "\n";
        assert(child.getLevel() == 2);
        for (int d = 0; d != Dimension; ++d) {
          assert(child.getCoordinates()[d] == 2 *((i >> d) % 2) +
                 (j >> d) % 2);
        }
        child.transformToParent();
      }
      child.transformToParent();
      assert(parent == child);
    }
  }
  // Neighbor
  {
    Index node, neighbor;
    for (int i = 0; i != 2 * Dimension; ++i) {
      assert(! hasNeighbor(node, i));
    }
    node.transformToChild(0);
    node.transformToChild(0);
    for (int d = 0; d != Dimension; ++d) {
      assert(! hasNeighbor(node, 2 * d));
      assert(hasNeighbor(node, 2 * d + 1));
      neighbor = node;
      neighbor.transformToNeighbor(2 * d + 1);
      assert(node != neighbor);
      assert(geom::areAdjacent(node, neighbor));
      neighbor.transformToNeighbor(2 * d);
      assert(node == neighbor);
      assert(! geom::areAdjacent(node, neighbor));

      neighbor.transformToNeighbor(2 * d + 1);
      neighbor.transformToChild(0);
      assert(geom::areAdjacent(node, neighbor));
      neighbor.transformToChild(0);
      assert(geom::areAdjacent(node, neighbor));
      neighbor.transformToParent();
      neighbor.transformToParent();
      neighbor.transformToNeighbor(2 * d);
      assert(node == neighbor);

      neighbor.transformToNeighbor(2 * d + 1);
      neighbor.transformToNeighbor(2 * d + 1);
      assert(! geom::areAdjacent(node, neighbor));
      neighbor.transformToNeighbor(2 * d);
      neighbor.transformToNeighbor(2 * d);
      assert(node == neighbor);

      int e = (d + 1) % Dimension;
      neighbor.transformToNeighbor(2 * d + 1);
      neighbor.transformToNeighbor(2 * e + 1);
      assert(! geom::areAdjacent(node, neighbor));
      neighbor.transformToNeighbor(2 * e);
      neighbor.transformToNeighbor(2 * d);
      assert(node == neighbor);
    }
  }
  // Location.
  {
    Index x;
    std::array<Coordinate, Dimension> location;
    geom::computeLocation(x, &location);
    // Initialize to zero.
    std::array<Coordinate, Dimension> position = {{}};
    assert(location == position);
    x.transformToChild(Index::NumberOfOrthants - 1);
    geom::computeLocation(x, &location);
    position += Coordinate(1 << (MaximumLevel - 1));
    assert(location == position);
    x.transformToChild(Index::NumberOfOrthants - 1);
    geom::computeLocation(x, &location);
    position += Coordinate(1 << (MaximumLevel - 2));
    assert(location == position);
  }
  // Length.
  {
    Index x;
    std::size_t length = 1 << MaximumLevel;
    assert(geom::computeLength(x) == length);
    x.transformToChild(Index::NumberOfOrthants - 1);
    length /= 2;
    assert(geom::computeLength(x) == length);
    x.transformToChild(Index::NumberOfOrthants - 1);
    length /= 2;
    assert(geom::computeLength(x) == length);
    while (x.canBeRefined()) {
      x.transformToChild(Index::NumberOfOrthants - 1);
    }
    assert(geom::computeLength(x) == 1);
  }
  // Separations.
  {
    Index a, b;
    int length = 1 << MaximumLevel;
    int pointDistance = 0;
    std::array<int, Dimension> separations;
    geom::computeSeparations(a, b, &separations);
    {
      std::array<int, Dimension> x;
      std::fill(x.begin(), x.end(), pointDistance - length);
      assert(separations == x);
    }

    a.transformToChild(0);
    b.transformToChild(Index::NumberOfOrthants - 1);
    length /= 2;
    pointDistance = length;
    geom::computeSeparations(a, b, &separations);
    for (int d = 0; d != Dimension; ++d) {
      assert(separations[0] == pointDistance - length);
    }

    while (length != 1) {
      a.transformToChild(0);
      b.transformToChild(0);
      length /= 2;
      geom::computeSeparations(a, b, &separations);
      for (int d = 0; d != Dimension; ++d) {
        assert(separations[0] == pointDistance - length);
      }
      geom::computeSeparations(b, a, &separations);
      for (int d = 0; d != Dimension; ++d) {
        assert(separations[0] == pointDistance - length);
      }
    }
  }
}
