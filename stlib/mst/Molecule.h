// -*- C++ -*-

/*!
  \file Molecule.h
  \brief The geometry of a molecule.
*/

#if !defined(__mst_Molecule_h__)
#define __mst_Molecule_h__

#include "stlib/mst/Atom.h"

#include "stlib/ads/iterator/TransformIterator.h"
#include "stlib/geom/orq/SortFirstDynamic.h"

#include <iostream>
#include <set>
#include <map>

#include <cassert>

namespace stlib
{
namespace mst
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! The geometry of a molecule.
template<typename T>
class Molecule
{
  //
  // Public types.
  //

public:

  //! The number type.
  typedef T Number;
  //! An atom.
  typedef geom::Ball<Number, 3> AtomType;
  //! A Cartesian point.
  typedef typename AtomType::Point Point;

  //
  // Private types.
  //

private:

  //! The container of atoms.
  typedef std::map<std::size_t, AtomType> AtomMap;
  typedef typename AtomMap::iterator IdentifierAndAtomIterator;

  //
  // More public types.
  //

public:

  //! A const iterator on pairs of identifiers and atoms.
  typedef typename AtomMap::const_iterator IdentifierAndAtomConstIterator;
  //! A const iterator on atom identifiers.
  typedef ads::TransformIterator < IdentifierAndAtomConstIterator,
          ads::Select1st<typename AtomMap::value_type> >
          IdentifierConstIterator;

  //
  // Nested classes.
  //

private:

  //! The multi-key accessor
  class MultiKeyAccessor :
    public std::unary_function<IdentifierAndAtomConstIterator, Point>
  {
  private:

    //! The base type.
    typedef std::unary_function<IdentifierAndAtomConstIterator, Point> Base;

  public:

    //! The argument type.
    typedef typename Base::argument_type argument_type;
    //! The result type.
    typedef typename Base::result_type result_type;

    //! Get the center of the atom.
    const result_type&
    operator()(const argument_type x) const
    {
      return x->second.center;
    }
  };

  //
  // More private types.
  //

  //! The ORQ data structure.
  typedef geom::SortFirstDynamic<3, MultiKeyAccessor> ORQ;
  //! A bounding box.
  typedef typename ORQ::BBox BBox;

  //
  // Data
  //

private:

  //! The map from identifiers to atoms.
  AtomMap _atoms;
  //! The ORQ data structure.
  ORQ _orq;
  //! The maximum atomic radius.
  Number _maxRadius;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{

  //! Default constructor.  Empty molecule.
  Molecule() :
    _atoms(),
    _orq(),
    _maxRadius(0) {}

  //! Copy constructor.
  Molecule(const Molecule& other) :
    _atoms(other._atoms),
    _orq(other._orq),
    _maxRadius(other._maxRadius) {}

  //! Assignment operator.
  Molecule&
  operator=(const Molecule& other)
  {
    // Avoid assignment to self.
    if (&other != this) {
      _atoms = other._atoms;
      _orq = other._orq;
      _maxRadius = other._maxRadius;
    }
    // Return *this so assignments can chain.
    return *this;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Get the beginning of the range of identifiers and atoms.
  IdentifierAndAtomConstIterator
  getIdentifiersAndAtomsBeginning() const
  {
    return _atoms.begin();
  }

  //! Get the end of the range of identifiers and atoms.
  IdentifierAndAtomConstIterator
  getIdentifiersAndAtomsEnd() const
  {
    return _atoms.end();
  }

  //! Get the beginning of the range of atom identifiers.
  IdentifierConstIterator
  getIdentifiersBeginning() const
  {
    return IdentifierConstIterator(_atoms.begin());
  }

  //! Get the end of the range of atom identifiers.
  IdentifierConstIterator
  getIdentifiersEnd() const
  {
    return IdentifierConstIterator(_atoms.end());
  }

  //! Get an identifier that is currently not being used.
  /*!
    \warning This is not efficient if the number of atoms is large.  I could
    add an implementation that uses an array with holes data structure.
    However, that would not be efficient if the identifiers were distributed
    over a large range.
  */
  int
  getUnusedIdentifier() const;

  //! Return true if this molecule has the specified atom.
  bool
  hasAtom(const std::size_t identifier) const
  {
    return _atoms.count(identifier) == 1;
  }

  //! Return the specified atom.
  const AtomType&
  getAtom(std::size_t identifier) const;

  //! Get the atoms that might clip the specified atom.
  template<typename IntOutputIterator, typename AtomOutputIterator>
  void
  getClippingAtoms(std::size_t identifier, IntOutputIterator identifiers,
                   AtomOutputIterator atoms) const;

  //! Get the atoms that are clipped by the specified atom.
  template<typename IntOutputIterator>
  void
  getClippedAtoms(std::size_t identifier, IntOutputIterator out) const;

  //! Compute the signed distance to the molecular surface.
  /*!
    The result is correct for positive distances, but may have the wrong
    magnitude for negative distances.  This is because I only compute
    distance to spheres, and not the intersecting curves.  Thus for negative
    distances, the magnitude may be too large.
  */
  Number
  computeSignedDistance(const Point& point) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Insert an atom with a specified identifier.
  void
  insert(std::size_t identifier, const AtomType& atom);

  //! Erase an atom.
  /*!
    \note The atom must be present in the molecule.  This is checked with an
    \c assert().  You can use hasAtom() to determine if the atom is present.
  */
  void
  erase(std::size_t identifier);

  //@}
  //--------------------------------------------------------------------------
  //! \name Equality.
  //@{

  //! Return true if the atom is equal to this one.
  bool
  isEqualTo(const Molecule& x) const;

  //@}
};


//! Return true if the atoms are equal.
template<typename T>
inline
bool
operator==(const Molecule<T>& x, const Molecule<T>& y)
{
  return x.isEqualTo(y);
}


//! Return true if the atoms are not equal.
template<typename T>
inline
bool
operator!=(const Molecule<T>& x, const Molecule<T>& y)
{
  return ! x == y;
}


} // namespace mst
}

#define __mst_Molecule_ipp__
#include "stlib/mst/Molecule.ipp"
#undef __mst_Molecule_ipp__

#endif
