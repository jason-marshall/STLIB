// -*- C++ -*-

#include "stlib/geom/mesh/iss/lor.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/laplacian.h"
#include "stlib/geom/mesh/iss/subdivide.h"

#include "stlib/ads/timer/Timer.h"

#include <iostream>

using namespace stlib;

typedef float Float;
const std::size_t Dimension = 3;
typedef geom::IndSimpSetIncAdj<Dimension, Dimension, Float> Mesh;

std::size_t
meaningless(const Mesh& mesh)
{
  std::size_t result =
    std::accumulate(mesh.incident.begin(), mesh.incident.end(), 0);
  for (std::size_t j = 0; j != mesh.vertices.size(); ++j) {
    for (std::size_t k = 0; k != Dimension; ++k) {
      result += std::size_t(mesh.vertices[j][k]);
    }
  }
  return result;
}

std::size_t
operations(const Mesh& mesh)
{
  ads::Timer timer;
  double elapsedTime;
  std::size_t result = 0;
  Mesh m = mesh;
  const std::size_t numTets = mesh.indexedSimplices.size();

  // Content.
  timer.tic();
  result += std::size_t(computeContent(m));
  elapsedTime = timer.toc();
  std::cout << ',' << 1e9 * elapsedTime / numTets;
  result += meaningless(m);

  // Laplacian smoothing.
  timer.tic();
  applyLaplacian(&m, 10);
  elapsedTime = timer.toc();
  std::cout << ',' << 1e9 * elapsedTime / (10 * numTets);
  result += meaningless(m);

  return result;
}

int
main()
{
  ads::Timer timer;
  double elapsedTime;
  std::size_t result = 0;

  // The column labels.
  std::cout <<
            "Tets,ContRand,LapRand,Axis,ContAxis,LapAxis,Mort8,ContMort8,LapMort8,Mort16,ContMort16,LapMort16,Mort32,ContMort32,LapMort32,Mort64,ContMort64,LapMort64\n";

  // Start with a tetrahedron.
  Mesh mesh;
  {
    const std::size_t numVertices = 4;
    float vertices[] = {0, 0, 0,
                        1, 0, 0,
                        0, 1, 0,
                        0, 0, 1
                       };
    const std::size_t numTets = 1;
    std::size_t tets[] = {0, 1, 2, 3};
    // Construct a mesh from vertices and tetrahedra.
    build(&mesh, numVertices, vertices, numTets, tets);
  }

  // Start with 4 levels of refinement.
  for (std::size_t i = 0; i != 4; ++i) {
    subdivide(mesh, &mesh);
  }

  for (std::size_t i = 0; i != 3; ++i) {
    // Refine the mesh.
    subdivide(mesh, &mesh);
    // Put the vertices and simplices in random order.
    randomOrder(&mesh);

    // The number of tets.
    const std::size_t numTets = mesh.indexedSimplices.size();
    std::cout << numTets;

    // Operations on the unordered mesh.
    result += operations(mesh);

    // Apply the axis order.
    timer.tic();
    axisOrder(&mesh);
    elapsedTime = timer.toc();
    std::cout << ',' << 1e9 * elapsedTime / numTets;

    // Operations on the axis-ordered mesh.
    result += operations(mesh);

    // Morton 8.
    randomOrder(&mesh);
    timer.tic();
    geom::mortonOrder<unsigned char>(&mesh);
    elapsedTime = timer.toc();
    std::cout << ',' << 1e9 * elapsedTime / numTets;

    // Operations on the morton-ordered mesh.
    result += operations(mesh);

    // Morton 16.
    randomOrder(&mesh);
    timer.tic();
    geom::mortonOrder<unsigned short>(&mesh);
    elapsedTime = timer.toc();
    std::cout << ',' << 1e9 * elapsedTime / numTets;

    // Operations on the morton-ordered mesh.
    result += operations(mesh);

    // Morton 32.
    randomOrder(&mesh);
    timer.tic();
    geom::mortonOrder<unsigned>(&mesh);
    elapsedTime = timer.toc();
    std::cout << ',' << 1e9 * elapsedTime / numTets;

    // Operations on the morton-ordered mesh.
    result += operations(mesh);

    // Morton 64.
    randomOrder(&mesh);
    timer.tic();
    geom::mortonOrder<std::size_t>(&mesh);
    elapsedTime = timer.toc();
    std::cout << ',' << 1e9 * elapsedTime / numTets;

    // Operations on the morton-ordered mesh.
    result += operations(mesh);

    std::cout << '\n';

  }

  std::cout << "\nMeaningless result = " << result << "\n";

  return 0;
}
