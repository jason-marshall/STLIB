// -*- C++ -*-

#include "stlib/ads/halfedge/HalfedgeDS.h"

#include "stlib/ads/halfedge/HDSVertex.h"
#include "stlib/ads/halfedge/HDSHalfedge.h"
#include "stlib/ads/halfedge/HDSFace.h"

#include <iostream>

using namespace stlib;

int
main()
{
  using namespace ads;

  // default constructor
  {
    HalfedgeDS<HDSVertex, HDSHalfedge, HDSFace> hds;
    std::cout << hds << '\n';
    assert(hds.is_valid());
    assert(hds.vertices_size() == 0);
    assert(hds.halfedges_size() == 0);
    assert(hds.faces_size() == 0);
  }

  // Make a triangle.  Reserve space.
  {
    typedef HalfedgeDS<HDSVertex, HDSHalfedge, HDSFace> HDS;
    typedef HDS::Vertex_handle Vertex_handle;
    typedef HDS::Halfedge_handle Halfedge_handle;
    typedef HDS::Face_handle Face_handle;
    typedef HDS::Face_Halfedge_const_circulator
    Face_Halfedge_const_circulator;

    HalfedgeDS<HDSVertex, HDSHalfedge, HDSFace> hds(3, 6, 1);
    assert(hds.is_valid());

    Vertex_handle v0 = hds.insert_vertex();
    Vertex_handle v1 = hds.insert_vertex();
    Vertex_handle v2 = hds.insert_vertex();

    Halfedge_handle h0 = hds.insert_halfedge();
    Halfedge_handle h1 = hds.insert_halfedge();
    Halfedge_handle h2 = hds.insert_halfedge();

    Face_handle f = hds.insert_face();

    v0->halfedge() = h0;
    v1->halfedge() = h1;
    v2->halfedge() = h2;

    h0->prev() = h2;
    h0->next() = h1;
    h0->vertex() = v0;
    h0->face() = f;

    h1->prev() = h0;
    h1->next() = h2;
    h1->vertex() = v1;
    h1->face() = f;

    h2->prev() = h1;
    h2->next() = h0;
    h2->vertex() = v2;
    h2->face() = f;

    f->halfedge() = h0;

    std::cout << "Triangle:\n" << hds << '\n';
    assert(hds.is_valid());

    // Count the halfedges of the face.
    {
      int count = 0;
      Face_Halfedge_const_circulator
      c = f->halfedges_begin(),
      begin = f->halfedges_begin();
      do {
        ++count;
        ++c;
      }
      while (c != begin);
      assert(count == 3);
    }
  }

  // CONTINUE: This does not work.
#if 0
  // Make a triangle. Don't reserve space.
  {
    typedef HalfedgeDS<HDSVertex, HDSHalfedge, HDSFace> HDS;
    typedef HDS::Vertex_iterator Vertex_iterator;
    typedef HDS::Vertex_handle Vertex_handle;
    typedef HDS::Halfedge_iterator Halfedge_iterator;
    typedef HDS::Halfedge_handle Halfedge_handle;
    typedef HDS::Face_handle Face_handle;

    HalfedgeDS<HDSVertex, HDSHalfedge, HDSFace> hds;
    assert(hds.is_valid());

    hds.insert_vertex();
    hds.insert_vertex();
    hds.insert_vertex();
    Vertex_iterator v = hds.vertices_begin();
    Vertex_handle v0 = v++;
    Vertex_handle v1 = v++;
    Vertex_handle v2 = v;

    hds.insert_halfedge();
    // CONTINUE: I think that the problem is here. The data structure is
    // not valid, and thus the resizing algorithms give erroneous results.
    hds.insert_halfedge();
    hds.insert_halfedge();
    Halfedge_iterator h = hds.halfedges_begin();
    Halfedge_handle h0 = h;
    h += 2;
    Halfedge_handle h1 = h;
    h += 2;
    Halfedge_handle h2 = h;

    Face_handle f = hds.insert_face();

    v0->halfedge() = h0;
    v1->halfedge() = h1;
    v2->halfedge() = h2;

    // CONTINUE Is this the correct value for opposite?
    h0->prev() = h2;
    h0->next() = h1;
    h0->vertex() = v0;
    h0->face() = f;

    h1->prev() = h0;
    h1->next() = h2;
    h1->vertex() = v1;
    h1->face() = f;

    h2->prev() = h1;
    h2->next() = h0;
    h2->vertex() = v2;
    h2->face() = f;

    f->halfedge() = h0;

    std::cout << "Triangle, don't reserve space:\n" << hds << '\n';
    assert(hds.is_valid());
  }
#endif

  return 0;
}
