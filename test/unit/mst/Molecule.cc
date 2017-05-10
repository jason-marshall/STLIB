// -*- C++ -*-

#include "stlib/mst/Molecule.h"

using namespace stlib;

#define PT ext::make_array<Number>

int
main()
{
  typedef double Number;
  typedef mst::Molecule<Number> Molecule;
  typedef Molecule::AtomType Atom;

  {
    // Default constructor.
    Molecule x;
    const Atom a = {{{1, 2, 3}}, 4};
    x.insert(0, a);
    assert(x.hasAtom(0));
    assert(! x.hasAtom(1));
    assert(x.getAtom(0) == a);

    {
      // Copy constructor.
      Molecule y(x);
      assert(y == x);
    }
    {
      // Assignment operator.
      Molecule y;
      y = x;
      assert(y == x);
    }
  }

  // getClippingAtoms.
  {
    Molecule x;
    {
      Atom a = {{{0, 0, 0}}, 0.9};
      x.insert(0, a);
      a.center[0] = 1;
      x.insert(1, a);
      a.center[0] = 2;
      x.insert(2, a);
    }

    std::vector<int> clippingIdentifiers;
    std::vector<Atom> clippingAtoms;

    x.getClippingAtoms(0, std::back_inserter(clippingIdentifiers),
                       std::back_inserter(clippingAtoms));
    assert(clippingIdentifiers.size() == 1);
    assert(clippingAtoms.size() == 1);
    clippingIdentifiers.clear();
    clippingAtoms.clear();

    x.getClippingAtoms(1, std::back_inserter(clippingIdentifiers),
                       std::back_inserter(clippingAtoms));
    assert(clippingIdentifiers.size() == 2);
    assert(clippingAtoms.size() == 2);
    clippingIdentifiers.clear();
    clippingAtoms.clear();

    x.getClippingAtoms(2, std::back_inserter(clippingIdentifiers),
                       std::back_inserter(clippingAtoms));
    assert(clippingIdentifiers.size() == 1);
    assert(clippingAtoms.size() == 1);
    clippingIdentifiers.clear();
    clippingAtoms.clear();
  }

  // getClippedAtoms.
  {
    Molecule x;
    {
      Atom a = {{{0, 0, 0}}, 0.9};
      x.insert(0, a);
      a.center[0] = 1;
      x.insert(1, a);
      a.center[0] = 2;
      x.insert(2, a);
    }

    std::vector<int> clipped;

    x.getClippedAtoms(0, std::back_inserter(clipped));
    assert(clipped.size() == 1);
    clipped.clear();

    x.getClippedAtoms(1, std::back_inserter(clipped));
    assert(clipped.size() == 2);
    clipped.clear();

    x.getClippedAtoms(2, std::back_inserter(clipped));
    assert(clipped.size() == 1);
    clipped.clear();
  }

  return 0;
}
