// -*- C++ -*-

#if !defined(__mst_Molecule_ipp__)
#error This file is an implementation detail of Molecule.
#endif

namespace stlib
{
namespace mst
{


//--------------------------------------------------------------------------
// Accessors.
//--------------------------------------------------------------------------


// Get an identifier that is currently not being used.
template<typename T>
inline
int
Molecule<T>::
getUnusedIdentifier() const
{
  IdentifierAndAtomConstIterator i = _atoms.begin();
  std::size_t n = 0;
  // Look for a hole in the sequence of identifiers.
  for (; i != _atoms.end(); ++i, ++n) {
    // If we find a hole.
    if (i->first != n) {
      break;
    }
  }
  // If we did not find a hole, we will return a number one greater than the
  // maximum identifier.
  return n;
}


// Return the specified atom.
template<typename T>
inline
const typename Molecule<T>::AtomType&
Molecule<T>::
getAtom(const std::size_t identifier) const
{
  // Find the atom.
  IdentifierAndAtomConstIterator i = _atoms.find(identifier);
  // Make sure that we found it.
  assert(i != _atoms.end());
  // Return the atom.
  return i->second;
}


// Get the atoms that might clip the specified atom.
template<typename T>
template<typename IntOutputIterator, typename AtomOutputIterator>
inline
void
Molecule<T>::
getClippingAtoms(std::size_t identifier, IntOutputIterator identifiers,
                 AtomOutputIterator atoms) const
{
  // Get the specified atom.
  const AtomType& atom = getAtom(identifier);

  // Get the atoms that are close.
  std::vector<IdentifierAndAtomConstIterator> closeAtoms;
  const Number offset = atom.radius + _maxRadius;
  BBox window = {atom.center - offset, atom.center + offset};
  _orq.computeWindowQuery(std::back_inserter(closeAtoms), window);

  // For each close atom.
  for (typename std::vector<IdentifierAndAtomConstIterator>::const_iterator
       i = closeAtoms.begin(); i != closeAtoms.end(); ++i) {
    // Except the specified atom.
    if ((*i)->first != identifier) {
      // If the atom clips the specified one.
      if (doesClip(atom, (*i)->second)) {
        // Record the identifiers and the atom.
        *identifiers++ = (*i)->first;
        *atoms++ = (*i)->second;
      }
    }
  }

  // CONTINUE
#if 0
  // For each atom.
  for (IdentifierAndAtomConstIterator i = _atoms.begin(); i != _atoms.end();
       ++i) {
    // Except the specified atom.
    if (i->first != identifier) {
      // If the atom clips the specified one.
      if (doesClip(atom, i->second)) {
        // Record the identifiers and the atom.
        *identifiers++ = i->first;
        *atoms++ = i->second;
      }
    }
  }
#endif
}


// Get the atom identifiers that the specified atom might clip.
template<typename T>
template<typename IntOutputIterator>
inline
void
Molecule<T>::
getClippedAtoms(const std::size_t identifier, IntOutputIterator out) const
{
  // Get the specified atom.
  const AtomType& atom = getAtom(identifier);

  // Get the atoms that are close.
  std::vector<IdentifierAndAtomConstIterator> closeAtoms;
  const Number offset = atom.radius + _maxRadius;
  BBox window = {atom.center - offset, atom.center + offset};
  _orq.computeWindowQuery(std::back_inserter(closeAtoms), window);

  // For each close atom.
  for (typename std::vector<IdentifierAndAtomConstIterator>::const_iterator
       i = closeAtoms.begin(); i != closeAtoms.end(); ++i) {
    // Except the specified atom.
    if ((*i)->first != identifier) {
      // If the specified atom clips this one.
      if (doesClip((*i)->second, atom)) {
        // Record the identifier.
        *out++ = (*i)->first;
      }
    }
  }

  // CONTINUE
#if 0
  // For each atom.
  for (IdentifierAndAtomConstIterator i = _atoms.begin(); i != _atoms.end();
       ++i) {
    // Except the specified atom.
    if (i->first != identifier) {
      // If the specified atom clips this one.
      if (doesClip(i->second, atom)) {
        // Record the identifier.
        *out++ = i->first;
      }
    }
  }
#endif
}



// Compute the signed distance to the molecular surface.
template<typename T>
inline
typename Molecule<T>::Number
Molecule<T>::
computeSignedDistance(const Point& point) const
{
  // Initialize with positive infinity.
  Number distance = std::numeric_limits<Number>::max();
  Number d;
  // For each atom.
  const IdentifierAndAtomConstIterator iEnd = _atoms.end();
  for (IdentifierAndAtomConstIterator i = _atoms.begin(); i != iEnd; ++i) {
    d = (geom::computeDistance(i->second.center, point) -
         i->second.radius);
    if (d < distance) {
      distance = d;
    }
  }
  return distance;
}



//--------------------------------------------------------------------------
// Manipulators.
//--------------------------------------------------------------------------


// Insert an atom.
template<typename T>
inline
void
Molecule<T>::
insert(const std::size_t identifier, const AtomType& atom)
{
  // Update the maximum atom radius.
  if (atom.radius > _maxRadius) {
    _maxRadius = atom.radius;
  }

  // Insert the atom.
  std::pair<IdentifierAndAtomIterator, bool> result =
    _atoms.insert(std::make_pair(identifier, atom));
  // Make sure it was inserted.
  assert(result.second);

  // Update the ORQ data structure.
  _orq.insert(result.first);
}


// Erase an atom.
template<typename T>
inline
void
Molecule<T>::
erase(const std::size_t identifier)
{
  // Find the atom.
  IdentifierAndAtomIterator i = _atoms.find(identifier);
  // Make sure that the atom exists.
  assert(i != _atoms.end());
  // Erase the atom.
  _atoms.erase(i);
  // Erase it from the ORQ data structure.
  _orq.erase(i);
}


//--------------------------------------------------------------------------
// Equality.
//--------------------------------------------------------------------------


// Return true if the atom is equal to this one.
template<typename T>
inline
bool
Molecule<T>::
isEqualTo(const Molecule& x) const
{
  if (_atoms != x._atoms) {
    return false;
  }
  if (_maxRadius != x._maxRadius) {
    return false;
  }
  return true;
}


} // namespace mst
}
