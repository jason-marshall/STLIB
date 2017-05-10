// -*- C++ -*-

#include "stlib/geom/mesh/iss/penetration.h"
#include "stlib/geom/mesh/iss/file_io.h"

#include <vector>
#include <iterator>
#include <sstream>

using namespace stlib;

void
triangle()
{
  typedef geom::IndSimpSetIncAdj<2, 2> Mesh;
  typedef Mesh::Vertex Vertex;
  typedef Mesh::IndexedSimplex IndexedSimplex;

  const double Epsilon = 10 * std::numeric_limits<double>::epsilon();

  std::vector<Vertex> vertices(3);
  vertices[0] = Vertex{{0.0, 0.0}};
  vertices[1] = Vertex{{1.0, 0.0}};
  vertices[2] = Vertex{{0.0, 1.0}};

  std::vector<IndexedSimplex> indexedSimplices(1);
  indexedSimplices[0] = IndexedSimplex{{0, 1, 2}};

  Mesh mesh(vertices, indexedSimplices);

  // Empty
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.1, 0.1}};
    assert(reportPenetrations(mesh, &vertex, &vertex,
                              std::back_inserter(penetrations)) == 0);
  }
  // Outside.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{ -1.0, 0.0}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 0);
  }
  // Outside, but inside the bounding box.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.6, 0.6}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 0);
  }
  // Inside, bottom edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.5, 0.1}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.5, 0.0}}) < Epsilon);
  }
  // Inside, left edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.1, 0.5}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.0, 0.5}}) < Epsilon);
  }
  // Inside, diagonal edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.4, 0.4}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.5, 0.5}}) < Epsilon);
  }
  // Vertices
  for (std::size_t i = 0; i != 3; ++i) {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = vertices[i];
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 vertex) < Epsilon);
  }
  // On bottom edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.5, 0.0}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.5, 0.0}}) < Epsilon);
  }
  // On left edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.0, 0.5}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.0, 0.5}}) < Epsilon);
  }
  // On diagonal edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.5, 0.5}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.5, 0.5}}) < Epsilon);
  }
  // Two inside.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    std::vector<Vertex> points;
    points.push_back(Vertex{{0.2, 0.1}});
    points.push_back(Vertex{{0.3, 0.1}});
    assert(reportPenetrations(mesh, points.begin(), points.end(),
                              std::back_inserter(penetrations)) == 2);
    assert(penetrations.size() == 2);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.2, 0.0}}) < Epsilon);
    assert(std::get<0>(penetrations[1]) == 1);
    assert(std::get<1>(penetrations[1]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[1]),
                                 Vertex{{0.3, 0.0}}) < Epsilon);
  }
}


void
square()
{
  typedef geom::IndSimpSetIncAdj<2, 2> Mesh;
  typedef Mesh::Vertex Vertex;
  typedef Mesh::IndexedSimplex IndexedSimplex;

  const double Epsilon = 10 * std::numeric_limits<double>::epsilon();

  std::vector<Vertex> vertices(4);
  vertices[0] = Vertex{{0.0, 0.0}};
  vertices[1] = Vertex{{1.0, 0.0}};
  vertices[2] = Vertex{{1.0, 1.0}};
  vertices[3] = Vertex{{0.0, 1.0}};

  std::vector<IndexedSimplex> indexedSimplices(2);
  indexedSimplices[0] = IndexedSimplex{{0, 1, 3}};
  indexedSimplices[1] = IndexedSimplex{{1, 2, 3}};

  Mesh mesh(vertices, indexedSimplices);

  // Outside.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{ -1.0, 0.0}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 0);
  }
  // Inside, bottom edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.5, 0.1}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.5, 0.0}}) < Epsilon);
  }
  // Inside, top edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.5, 0.9}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 1);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.5, 1.0}}) < Epsilon);
  }
  // Inside, left edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.1, 0.5}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.0, 0.5}}) < Epsilon);
  }
  // Inside, right edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.9, 0.5}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 1);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{1.0, 0.5}}) < Epsilon);
  }
  // Vertices
  for (std::size_t i = 0; i != 4; ++i) {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = vertices[i];
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    //assert(std::get<1>(penetrations[0]) == );
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 vertex) < Epsilon);
  }
  // On bottom edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.5, 0.0}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.5, 0.0}}) < Epsilon);
  }
  // On top edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.5, 1.0}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 1);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.5, 1.0}}) < Epsilon);
  }
  // On left edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.0, 0.5}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.0, 0.5}}) < Epsilon);
  }
  // On right edge.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{1.0, 0.5}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 1);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{1.0, 0.5}}) < Epsilon);
  }
  // Two inside.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    std::vector<Vertex> points;
    points.push_back(Vertex{{0.2, 0.1}});
    points.push_back(Vertex{{0.3, 0.1}});
    assert(reportPenetrations(mesh, points.begin(), points.end(),
                              std::back_inserter(penetrations)) == 2);
    assert(penetrations.size() == 2);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.2, 0.0}}) < Epsilon);
    assert(std::get<0>(penetrations[1]) == 1);
    assert(std::get<1>(penetrations[1]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[1]),
                                 Vertex{{0.3, 0.0}}) < Epsilon);
  }
}


void
tet()
{
  typedef geom::IndSimpSetIncAdj<3, 3> Mesh;
  typedef Mesh::Vertex Vertex;
  typedef Mesh::IndexedSimplex IndexedSimplex;

  const double Epsilon = 10 * std::numeric_limits<double>::epsilon();

  std::vector<Vertex> vertices(4);
  vertices[0] = Vertex{{0.0, 0.0, 0.0}};
  vertices[1] = Vertex{{1.0, 0.0, 0.0}};
  vertices[2] = Vertex{{0.0, 1.0, 0.0}};
  vertices[3] = Vertex{{0.0, 0.0, 1.0}};

  std::vector<IndexedSimplex> indexedSimplices(1);
  indexedSimplices[0] = IndexedSimplex{{0, 1, 2, 3}};

  Mesh mesh(vertices, indexedSimplices);

  // Outside.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{ -1.0, 0.0, 0.0}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 0);
  }
  // Outside, but inside the bounding box.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.6, 0.6, 0.6}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 0);
  }
  // Inside, close to face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.25, 0.25, 0.1}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.25, 0.25, 0.0}}) < Epsilon);
  }
  // Inside, close to face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.25, 0.1, 0.25}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.25, 0.0, 0.25}}) < Epsilon);
  }
  // Inside, close to face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.1, 0.25, 0.25}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.0, 0.25, 0.25}}) < Epsilon);
  }
  // Inside, close to face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.3, 0.3, 0.3}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{1. / 3, 1. / 3, 1. / 3}}) < Epsilon);
  }
  // Vertices
  for (std::size_t i = 0; i != 4; ++i) {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = vertices[i];
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 vertex) < Epsilon);
  }
  // On face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.25, 0.25, 0.0}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.25, 0.25, 0.0}}) < Epsilon);
  }
  // On face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.25, 0.0, 0.25}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.25, 0.0, 0.25}}) < Epsilon);
  }
  // On face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.0, 0.25, 0.25}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.0, 0.25, 0.25}}) < Epsilon);
  }
  // On face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{1. / 3 - Epsilon, 1. / 3 - Epsilon, 1. / 3 - Epsilon}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{1. / 3, 1. / 3, 1. / 3}}) < Epsilon);
  }
  // Two inside.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    std::vector<Vertex> points;
    points.push_back(Vertex{{0.2, 0.2, 0.1}});
    points.push_back(Vertex{{0.3, 0.3, 0.1}});
    assert(reportPenetrations(mesh, points.begin(), points.end(),
                              std::back_inserter(penetrations)) == 2);
    assert(penetrations.size() == 2);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.2, 0.2, 0.0}}) < Epsilon);
    assert(std::get<0>(penetrations[1]) == 1);
    assert(std::get<1>(penetrations[1]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[1]),
                                 Vertex{{0.3, 0.3, 0.0}}) < Epsilon);
  }
}


void
cube()
{
  typedef geom::IndSimpSetIncAdj<3, 3> Mesh;
  typedef Mesh::Vertex Vertex;

  const double Epsilon = 10 * std::numeric_limits<double>::epsilon();

  const char data[] =
    "3 3 8 0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 1 5 0 1 3 4 2 3 1 6 5 4 6 1 7 6 4 3 1 3 4 6";
  std::istringstream in(data);

  Mesh mesh;
  geom::readAscii(in, &mesh);
  assert(mesh.vertices.size() == 8);
  assert(mesh.indexedSimplices.size() == 5);

  // Outside.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{ -1.0, 0.0, 0.0}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 0);
  }
  // Inside, close to face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.25, 0.25, 0.1}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.25, 0.25, 0.0}}) < Epsilon);
  }
  // Inside, close to face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.25, 0.1, 0.25}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.25, 0.0, 0.25}}) < Epsilon);
  }
  // Inside, close to face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.1, 0.25, 0.25}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.0, 0.25, 0.25}}) < Epsilon);
  }
  // Inside, close to face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.5, 0.5, 0.9}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    //assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.5, 0.5, 1.0}}) < Epsilon);
  }
  // Vertices
  for (std::size_t i = 0; i != mesh.vertices.size(); ++i) {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = mesh.vertices[i];
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    //assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 vertex) < Epsilon);
  }
  // On face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.25, 0.25, 0.0}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.25, 0.25, 0.0}}) < Epsilon);
  }
  // On face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.25, 0.0, 0.25}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.25, 0.0, 0.25}}) < Epsilon);
  }
  // On face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.0, 0.25, 0.25}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.0, 0.25, 0.25}}) < Epsilon);
  }
  // On face.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    Vertex vertex = {{0.5, 0.5, 1.0}};
    assert(reportPenetrations(mesh, &vertex, &vertex + 1,
                              std::back_inserter(penetrations)) == 1);
    assert(penetrations.size() == 1);
    assert(std::get<0>(penetrations[0]) == 0);
    //assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.5, 0.5, 1.0}}) < Epsilon);
  }
  // Two inside.
  {
    std::vector<std::tuple<std::size_t, std::size_t, Vertex> > penetrations;
    std::vector<Vertex> points;
    points.push_back(Vertex{{0.2, 0.2, 0.1}});
    points.push_back(Vertex{{0.3, 0.3, 0.1}});
    assert(reportPenetrations(mesh, points.begin(), points.end(),
                              std::back_inserter(penetrations)) == 2);
    assert(penetrations.size() == 2);
    assert(std::get<0>(penetrations[0]) == 0);
    assert(std::get<1>(penetrations[0]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[0]),
                                 Vertex{{0.2, 0.2, 0.0}}) < Epsilon);
    assert(std::get<0>(penetrations[1]) == 1);
    assert(std::get<1>(penetrations[1]) == 0);
    assert(geom::computeDistance(std::get<2>(penetrations[1]),
                                 Vertex{{0.3, 0.3, 0.0}}) < Epsilon);
  }
}


int
main()
{
  // Test reportPenetrations.
  triangle();
  square();
  tet();
  cube();

  // Test maximumRelativePenetration.
  {
    typedef geom::IndSimpSetIncAdj<2, 2> Mesh;
    typedef Mesh::Vertex Vertex;
    typedef Mesh::IndexedSimplex IndexedSimplex;
    Vertex vertices[] = {{{0, 0}},
      {{1, 0}},
      {{0, 1}}
    };
    IndexedSimplex indexedSimplices[] = {{{0, 1, 2}}};
    Mesh mesh;
    build(&mesh, sizeof(vertices) / sizeof(Vertex), vertices,
          sizeof(indexedSimplices) / sizeof(IndexedSimplex),
          indexedSimplices);
    assert(maximumRelativePenetration(mesh) == 0);
  }
  {
    typedef geom::IndSimpSetIncAdj<2, 2> Mesh;
    typedef Mesh::Vertex Vertex;
    typedef Mesh::IndexedSimplex IndexedSimplex;
    Vertex vertices[] = {{{0, 0}},
      {{1, 0}},
      {{0, 1}},
      {{1, 0}},
      {{2, 0}},
      {{1, 1}}
    };
    IndexedSimplex indexedSimplices[] = {{{0, 1, 2}}, {{3, 4, 5}}};
    Mesh mesh;
    build(&mesh, sizeof(vertices) / sizeof(Vertex), vertices,
          sizeof(indexedSimplices) / sizeof(IndexedSimplex),
          indexedSimplices);
    assert(maximumRelativePenetration(mesh) == 0);
  }
  {
    typedef geom::IndSimpSetIncAdj<2, 2> Mesh;
    typedef Mesh::Vertex Vertex;
    typedef Mesh::IndexedSimplex IndexedSimplex;
    Vertex vertices[] = {{{0, 0}},
      {{1, 0}},
      {{0, 1}},
      {{0.1, -0.9}},
      {{1.1, -0.9}},
      {{0.1, 0.1}}
    };
    IndexedSimplex indexedSimplices[] = {{{0, 1, 2}}, {{3, 4, 5}}};
    Mesh mesh;
    build(&mesh, sizeof(vertices) / sizeof(Vertex), vertices,
          sizeof(indexedSimplices) / sizeof(IndexedSimplex),
          indexedSimplices);
    const double Value = 0.1 / std::sqrt(2.);
    const double Eps = 10 * Value * std::numeric_limits<double>::epsilon();
    assert(std::abs(maximumRelativePenetration(mesh) - Value) < Eps);
  }
  {
    typedef geom::IndSimpSetIncAdj<3, 3> Mesh;
    typedef Mesh::Vertex Vertex;
    typedef Mesh::IndexedSimplex IndexedSimplex;
    Vertex vertices[] = {{{0, 0, 0}},
      {{1, 0, 0}},
      {{0, 1, 0}},
      {{0, 0, 1}},
      {{0, 0, 0}},
      {{1, 0, 0}},
      {{0, 1, 0}},
      {{0, 0, 1}}
    };
    IndexedSimplex indexedSimplices[] = {{{0, 1, 2, 3}}, {{4, 5, 6, 7}}};
    Mesh mesh;
    build(&mesh, sizeof(vertices) / sizeof(Vertex), vertices,
          sizeof(indexedSimplices) / sizeof(IndexedSimplex),
          indexedSimplices);
    assert(maximumRelativePenetration(mesh) == 0);
  }
  {
    typedef geom::IndSimpSetIncAdj<3, 3> Mesh;
    typedef Mesh::Vertex Vertex;
    typedef Mesh::IndexedSimplex IndexedSimplex;
    Vertex vertices[] = {{{0, 0, 0}},
      {{1, 0, 0}},
      {{0, 1, 0}},
      {{0, 0, 1}},
      {{0.1, 0.1, -0.9}},
      {{1.1, 0.1, -0.9}},
      {{0.1, 1.1, -0.9}},
      {{0.1, 0.1, 0.1}}
    };
    IndexedSimplex indexedSimplices[] = {{{0, 1, 2, 3}}, {{4, 5, 6, 7}}};
    Mesh mesh;
    build(&mesh, sizeof(vertices) / sizeof(Vertex), vertices,
          sizeof(indexedSimplices) / sizeof(IndexedSimplex),
          indexedSimplices);
    const double Value = 0.1 / std::sqrt(2.);
    const double Eps = 10 * Value * std::numeric_limits<double>::epsilon();
    assert(std::abs(maximumRelativePenetration(mesh) - Value) < Eps);
  }

  return 0;
}
