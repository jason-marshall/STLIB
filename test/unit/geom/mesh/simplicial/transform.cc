// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/transform.h"
#include "stlib/geom/mesh/iss/build.h"

#include "stlib/ads/functor.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // Transform.
  //
  {
    typedef geom::IndSimpSet<2, 2> ISS;
    typedef geom::SimpMeshRed<2, 2> SMR;
    typedef SMR::VertexIterator VertexIterator;
    typedef std::array<double, 2> Pt;

    const std::size_t numVertices = 5;
    double vertices[] = {0, 0,
                         1, 0,
                         0, 1,
                         -1, 0,
                         0, -1
                        };
    const std::size_t numSimplices = 4;
    std::size_t indexedSimplices[] = {0, 1, 2,
                                      0, 2, 3,
                                      0, 3, 4,
                                      0, 4, 1
                                     };

    ISS iss;
    build(&iss, numVertices, vertices, numSimplices, indexedSimplices);
    SMR mesh(iss);
    const Pt value = {{2, 3}};
    geom::transform(&mesh, ads::constructUnaryConstant<Pt, Pt>(value));
    for (VertexIterator i = mesh.getVerticesBeginning();
         i != mesh.getVerticesEnd(); ++i) {
      assert(*i == value);
    }
  }
  {
    typedef geom::IndSimpSet<2, 2> ISS;
    typedef geom::SimpMeshRed<2, 2> SMR;
    typedef SMR::Node Node;
    typedef SMR::NodeIterator NodeIterator;
    typedef SMR::VertexIterator VertexIterator;
    typedef std::array<double, 2> Pt;

    const std::size_t numVertices = 5;
    double vertices[] = {
      0, 0,
      1, 0,
      0, 1,
      -1, 0,
      0, -1
    };
    const std::size_t numSimplices = 4;
    std::size_t indexedSimplices[] = {
      0, 1, 2,
      0, 2, 3,
      0, 3, 4,
      0, 4, 1
    };

    ISS iss;
    build(&iss, numVertices, vertices, numSimplices, indexedSimplices);
    SMR mesh(iss);
    const Pt value = {{2, 3}};
    std::vector<Node*> nodes;
    for (NodeIterator i = mesh.getNodesBeginning(); i != mesh.getNodesEnd();
         ++i) {
      nodes.push_back(&*i);
    }
    geom::transformNodes<SMR>(nodes.begin(), nodes.end(),
                              ads::constructUnaryConstant<Pt, Pt>(value));
    for (VertexIterator i = mesh.getVerticesBeginning();
         i != mesh.getVerticesEnd(); ++i) {
      assert(*i == value);
    }
  }

  return 0;
}
