// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/SmrCell.h"

#include "stlib/geom/mesh/simplicial/insert.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace geom;

  {
    typedef SimpMeshRed<3> SMR;
    typedef SMR::Node Node;
    typedef SMR::Cell Cell;
    typedef SMR::CellIterator CI;
    typedef SMR::Vertex VT;

    {
      Node v;
    }
    {
      VT p = {{0, 0, 0}};
      CI c;
      Node v(p, 0);
    }

    {
      Cell c;
      //Cell c;
    }
    {
      Node* v0 = 0;
      Node* v1 = 0;
      Node* v2 = 0;
      Node* v3 = 0;
      Cell* c0 = 0;
      Cell* c1 = 0;
      Cell* c2 = 0;
      Cell* c3 = 0;
      Cell c(v0, v1, v2, v3, c0, c1, c2, c3);
    }

    {
      // Make a cell.
      SMR x;
      std::array<Node*, 4> v;
      for (std::size_t i = 0; i != 4; ++i) {
        v[i] = &*x.insertNode();
      }
      CI c = geom::insertCell(&x, v[0], v[1], v[2], v[3]);

      // Add neighbors.
      std::array<Node*, 4> nv;
      for (std::size_t i = 0; i != 4; ++i) {
        nv[i] = &*x.insertNode();
      }
      std::array<Cell*, 4> nc;
      Cell* z = 0;
      nc[0] = &*geom::insertCell(&x, nv[0], v[1], v[2], v[3], &*c, z, z, z);
      nc[1] = &*geom::insertCell(&x, v[0], nv[1], v[2], v[3], z, &*c, z, z);
      nc[2] = &*geom::insertCell(&x, v[0], v[1], nv[2], v[3], z, z, &*c, z);
      nc[3] = &*geom::insertCell(&x, v[0], v[1], v[2], nv[3], z, z, z, &*c);
      for (std::size_t i = 0; i != 4; ++i) {
        c->setNeighbor(i, nc[i]);
      }

      // Self pointer.
      assert(c->getSelf() == c);

      //
      // Vertex tests.
      //

      for (std::size_t i = 0; i != 4; ++i) {
        assert(c->getNode(i) == v[i]);
      }

      for (std::size_t i = 0; i != 4; ++i) {
        assert(c->getIndex(v[i]) == i);
      }

      std::array<const Node*, 4> vc;
      for (std::size_t i = 0; i != 4; ++i) {
        vc[i] = v[i];
      }

      std::size_t j;
      for (std::size_t i = 0; i != 4; ++i) {
        assert(c->hasNode(v[i]));
        c->hasNode(v[i], &j);
        assert(i == j);
      }

      //
      // Face tests.
      //

      // CONTINUE
#if 0
      CFT f;
      for (std::size_t i = 0; i != 4; ++i) {
        c->getFace(i, &f);
        assert(c->getIndex(f) == i);
      }
#endif

      //
      // Neighbor tests.
      //

      for (std::size_t i = 0; i != 4; ++i) {
        assert(c->getNeighbor(i) == nc[i]);
      }

      for (std::size_t i = 0; i != 4; ++i) {
        assert(c->getIndex(nc[i]) == i);
      }

      for (std::size_t i = 0; i != 4; ++i) {
        assert(c->hasNeighbor(nc[i]));
      }
      assert(! c->hasNeighbor(&*c));

      for (std::size_t i = 0; i != 4; ++i) {
        c->hasNeighbor(nc[i], &j);
        assert(i == j);
      }
      assert(! c->hasNeighbor(&*c, &j));

      for (std::size_t i = 0; i != 4; ++i) {
        assert(c->getMirrorIndex(i) == i);
      }

      for (std::size_t i = 0; i != 4; ++i) {
        assert(c->getMirrorVertex(i) == nv[i]);
      }

    }
  }

  return 0;
}
