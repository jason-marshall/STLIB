// -*- C++ -*-

#include "stlib/cpt/State.h"

#include <iostream>

using namespace stlib;

int
main()
{
  //-------------------------------------------------------------------------
  // 3-D
  //-------------------------------------------------------------------------
  {
    cpt::State<3> state;
    std::cout << "\n3-D, uninitialized:\n";
    state.displayInformation(std::cout);
  }
  {
    // Use multi-arrays.
    const std::size_t N = 3;
    typedef cpt::State<N> State;
    typedef State::Point Point;
    typedef State::SizeList SizeList;
    typedef State::IndexList IndexList;

    State state;

    geom::BBox<double, N> domain = {{{0, 0, 0}}, {{1, 1, 1}}};
    const SizeList extents = {{10, 10, 10}};
    const IndexList bases = {{}};
    state.setParameters(domain, 2.0);
    state.setLattice(extents, domain);
    state.insertGrid(extents, bases, true, true, true);

    {
      std::vector<Point> vertices;
      vertices.push_back(Point{{0.25, 0.25, 0.5}});
      vertices.push_back(Point{{0.75, 0.25, 0.5}});
      vertices.push_back(Point{{0.5, 0.75, 0.5}});
      std::vector<std::array<std::size_t, 3> > faces;
      faces.push_back(std::array<std::size_t, 3>{{0, 1, 2}});
      state.setBRepWithNoClipping(vertices, faces);

      std::cout << "\n3-D, triangle, signed, multi-arrays:\n";
      std::pair<std::size_t, std::size_t> counts =
        state.computeClosestPointTransform();
      std::cout << "Number of scan converted points = " << counts.first << '\n'
                << "Number of distances computed = " << counts.second << '\n';
      state.displayInformation(std::cout);
      // Avoid the error messages.
      //assert(! state.areGridsValid());

      std::cout << "\n3-D, triangle, unsigned, multi-arrays:\n";
      counts = state.computeClosestPointTransformUnsigned();
      std::cout << "Number of scan converted points = " << counts.first << '\n'
                << "Number of distances computed = " << counts.second << '\n';
      state.displayInformation(std::cout);
      assert(state.areGridsValidUnsigned());
    }
    {
      std::vector<Point> vertices;
      vertices.push_back(Point{{0, 0, 0}});
      vertices.push_back(Point{{1, 0, 0}});
      vertices.push_back(Point{{0, 1, 0}});
      vertices.push_back(Point{{0, 0, 1}});
      std::vector<std::array<std::size_t, 3> > faces;
      faces.push_back(std::array<std::size_t, 3>{{0, 2, 1}});
      faces.push_back(std::array<std::size_t, 3>{{0, 3, 2}});
      faces.push_back(std::array<std::size_t, 3>{{0, 1, 3}});
      faces.push_back(std::array<std::size_t, 3>{{1, 2, 3}});
      state.setBRepWithNoClipping(vertices, faces);

      std::cout << "\n3-D, tetrahedron, signed, multi-arrays:\n";
      std::pair<int, int> counts = state.computeClosestPointTransform();
      std::cout << "Number of scan converted points = " << counts.first << '\n'
                << "Number of distances computed = " << counts.second << '\n';
      state.displayInformation(std::cout);
      assert(state.areGridsValid());

      std::cout << "\n3-D, tetrahedron, unsigned, multi-arrays:\n";
      counts = state.computeClosestPointTransformUnsigned();
      std::cout << "Number of scan converted points = " << counts.first << '\n'
                << "Number of distances computed = " << counts.second << '\n';
      state.displayInformation(std::cout);
      assert(state.areGridsValidUnsigned());
    }
  }
// CONTINUE
#if 0
  //-------------------------------------------------------------------------
  // 2-D
  //-------------------------------------------------------------------------
  {
    cpt::State<2> state;
    std::cout << "\n2-D, uninitialized:\n";
    state.displayInformation(std::cout);
  } {
    // Use ads arrays.
    cpt::State<2> state;

    state.initializeParameters(1.0, true, 0, 0);

    ads::Array<2> distance(10, 10);
    ads::Array< 2, ads::FixedArray<2> > gradientOfDistance(10, 10);
    ads::Array< 2, ads::FixedArray<2> > closestPoint(10, 10);
    ads::Array<2, int> closestFace(10, 10);
    state.initializeGrid(geom::BBox<2>(Point{{0, 0}}, Point{{1, 1}}), &distance,
                         &gradientOfDistance, &closestPoint, &closestFace);

    {
      // Line segment.
      std::vector<ads::FixedArray<2> > vertices;
      vertices.push_back(ads::FixedArray<2>(0.75, 0.5));
      vertices.push_back(ads::FixedArray<2>(0.25, 0.5));
      std::vector<ads::FixedArray<2, int> > faces;
      faces.push_back(ads::FixedArray<2, int>(0, 1));
      state.initializeBRepWithNoClipping(vertices.begin(), vertices.end(),
                                         faces.begin(), faces.end());

      std::cout << "\n2-D, signed distance, line segment, ADS arrays:\n";
      std::pair<int, int> counts = state.computeClosestPointTransform();
      std::cout << "Number of scan converted points = " << counts.first
                << '\n'
                << "Number of distances computed = " << counts.second
                << '\n';
      state.displayInformation(std::cout);
      assert(state.areGridsValid());

      std::cout << "\n2-D, unsigned distance, line segment, ADS arrays:\n";
      counts = state.computeClosestPointTransformUnsigned();
      std::cout << "Number of scan converted points = " << counts.first
                << '\n'
                << "Number of distances computed = " << counts.second
                << '\n';
      state.displayInformation(std::cout);
      assert(state.areGridsValidUnsigned());
    }
    {
      // Triangle.
      std::vector< ads::FixedArray<2> > vertices;
      vertices.push_back(ads::FixedArray<2>(0, 0));
      vertices.push_back(ads::FixedArray<2>(1, 0));
      vertices.push_back(ads::FixedArray<2>(0, 1));
      std::vector< ads::FixedArray<2, int> > faces;
      faces.push_back(ads::FixedArray<2, int>(0, 1));
      faces.push_back(ads::FixedArray<2, int>(1, 2));
      faces.push_back(ads::FixedArray<2, int>(2, 0));
      state.initializeBRepWithNoClipping(vertices.begin(), vertices.end(),
                                         faces.begin(), faces.end());

      std::cout << "\n2-D, signed distance, triangle, ADS arrays:\n";
      std::pair<int, int> counts = state.computeClosestPointTransform();
      std::cout << "Number of scan converted points = " << counts.first
                << '\n'
                << "Number of distances computed = " << counts.second
                << '\n';
      state.displayInformation(std::cout);
      assert(state.areGridsValid());

      std::cout << "\n2-D, unsigned distance, triangle, ADS arrays:\n";
      counts = state.computeClosestPointTransformUnsigned();
      std::cout << "Number of scan converted points = " << counts.first
                << '\n'
                << "Number of distances computed = " << counts.second
                << '\n';
      state.displayInformation(std::cout);
      assert(state.areGridsValidUnsigned());
    }
  } {
    // Use C arrays.
    cpt::State<2> state;

    state.initializeParameters(1.0, true, 0, 0);

    const double domain[4] = {0, 0, 1, 1};
    const int gridDimensions[2] = {10, 10};
    const int gridSize = 100;
    double* distance = new double[gridSize];
    double* gradientOfDistance = new double[2 * gridSize];
    double* closestPoint = new double[2 * gridSize];
    int* closestFace = new int[gridSize];
    state.initializeGrid(domain, gridDimensions, distance,
                         gradientOfDistance, closestPoint, closestFace);

    const int verticesSize = 3;
    const double vertices[2 * verticesSize] = {
      0, 0,
      1, 0,
      0, 1
    };
    const int facesSize = 3;
    const int faces[2 * facesSize] = {
      0, 1,
      1, 2,
      2, 0
    };
    state.initializeBRepWithNoClipping(verticesSize, vertices,
                                       facesSize, faces);

    std::cout << "\n2-D, signed distance, initialized with C arrays:\n";
    std::pair<int, int> counts = state.computeClosestPointTransform();
    std::cout << "Number of scan converted points = " << counts.first
              << '\n'
              << "Number of distances computed = " << counts.second
              << '\n';
    state.displayInformation(std::cout);
    assert(state.areGridsValid());

    std::cout << "\n2-D, unsigned distance, initialized with C arrays:\n";
    counts = state.computeClosestPointTransformUnsigned();
    std::cout << "Number of scan converted points = " << counts.first
              << '\n'
              << "Number of distances computed = " << counts.second
              << '\n';
    state.displayInformation(std::cout);
    assert(state.areGridsValidUnsigned());

    state.initializeGrid(domain, gridDimensions, distance, 0, 0, 0);

    std::cout << "\n2-D, initialized with C arrays, only signed distance:\n";
    counts = state.computeClosestPointTransform();
    std::cout << "Number of scan converted points = " << counts.first
              << '\n'
              << "Number of distances computed = " << counts.second
              << '\n';
    state.displayInformation(std::cout);
    assert(state.areGridsValid());

    std::cout << "\n2-D, initialized with C arrays, only unsigned distance:\n";
    counts = state.computeClosestPointTransformUnsigned();
    std::cout << "Number of scan converted points = " << counts.first
              << '\n'
              << "Number of distances computed = " << counts.second
              << '\n';
    state.displayInformation(std::cout);
    assert(state.areGridsValidUnsigned());

    delete[] distance;
    delete[] gradientOfDistance;
    delete[] closestPoint;
    delete[] closestFace;
  }

  //-------------------------------------------------------------------------
  // 1-D
  //-------------------------------------------------------------------------
  {
    cpt::State<1> state;
    std::cout << "\n1-D, uninitialized:\n";
    state.displayInformation(std::cout);
  } {
    // Use ads arrays.
    cpt::State<1> state;

    state.initializeParameters(1.0);

    ads::Array<1> distance(10);
    ads::Array< 1, ads::FixedArray<1> > gradientOfDistance(10);
    ads::Array< 1, ads::FixedArray<1> > closestPoint(10);
    ads::Array<1, int> closestFace(10);
    state.initializeGrid(geom::BBox<1>(Point{{0}}, Point{{1}}), distance,
                         gradientOfDistance, closestPoint, closestFace);

    std::vector<double> faces;
    faces.push_back(0.0);
    faces.push_back(0.5);
    faces.push_back(1.0);
    std::vector<int> orientations;
    orientations.push_back(1);
    orientations.push_back(-1);
    orientations.push_back(1);
    state.initializeBRepWithNoClipping(faces.begin(), faces.end(),
                                       orientations.begin(), orientations.end());

    std::cout << "\n1-D, signed distance, initialized with ADS arrays:\n";
    std::pair<int, int> counts = state.computeClosestPointTransform();
    std::cout << "Number of scan converted points = " << counts.first
              << '\n'
              << "Number of distances computed = " << counts.second
              << '\n';
    state.displayInformation(std::cout);
    assert(state.areGridsValid());

    std::cout << "\n1-D, unsigned distance, initialized with ADS arrays:\n";
    counts = state.computeClosestPointTransformUnsigned();
    std::cout << "Number of scan converted points = " << counts.first
              << '\n'
              << "Number of distances computed = " << counts.second
              << '\n';
    state.displayInformation(std::cout);
    assert(state.areGridsValidUnsigned());
  } {
    // Use C arrays.
    cpt::State<1> state;

    state.initializeParameters(1.0);

    const double domain[2] = {0, 1};
    const int gridDimensions[1] = {10};
    const int gridSize = 10;
    double* distance = new double[gridSize];
    double* gradientOfDistance = new double[1 * gridSize];
    double* closestPoint = new double[1 * gridSize];
    int* closestFace = new int[gridSize];
    state.initializeGrid(domain, gridDimensions, distance,
                         gradientOfDistance, closestPoint, closestFace);

    const int facesSize = 3;
    const double faces[facesSize] = { 0.0, 0.5, 1.0 };
    const int orientations[facesSize] = { 1, -1, 1 };
    state.initializeBRepWithNoClipping(facesSize, faces, orientations);

    std::cout << "\n1-D, signed distance, initialized with C arrays:\n";
    std::pair<int, int> counts = state.computeClosestPointTransform();
    std::cout << "Number of scan converted points = " << counts.first
              << '\n'
              << "Number of distances computed = " << counts.second
              << '\n';
    state.displayInformation(std::cout);
    assert(state.areGridsValid());

    std::cout << "\n1-D, unsigned distance, initialized with C arrays:\n";
    counts = state.computeClosestPointTransformUnsigned();
    std::cout << "Number of scan converted points = " << counts.first
              << '\n'
              << "Number of distances computed = " << counts.second
              << '\n';
    state.displayInformation(std::cout);
    assert(state.areGridsValidUnsigned());

    state.initializeGrid(domain, gridDimensions, distance, 0, 0, 0);

    std::cout << "\n1-D, signed distance, initialized with C arrays, "
              << "only distance:\n";
    counts = state.computeClosestPointTransform();
    std::cout << "Number of scan converted points = " << counts.first
              << '\n'
              << "Number of distances computed = " << counts.second
              << '\n';
    state.displayInformation(std::cout);
    assert(state.areGridsValid());

    std::cout << "\n1-D, unsigned distance, initialized with C arrays, "
              << "only distance:\n";
    counts = state.computeClosestPointTransformUnsigned();
    std::cout << "Number of scan converted points = " << counts.first
              << '\n'
              << "Number of distances computed = " << counts.second
              << '\n';
    state.displayInformation(std::cout);
    assert(state.areGridsValidUnsigned());

    delete[] distance;
    delete[] gradientOfDistance;
    delete[] closestPoint;
    delete[] closestFace;
  }
#endif

  return 0;
}
