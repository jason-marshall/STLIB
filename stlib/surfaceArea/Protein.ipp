// -*- C++ -*-

#if !defined(__surfaceArea_Protein_ipp__)
#error This file is an implementation detail of Protein.
#endif

namespace stlib
{
namespace surfaceArea
{


template<typename Float>
inline
void
Protein<Float>::
meshReferenceAtoms()
{
  assert(_referenceMeshes.size() == _referenceRadii.size());
  assert(_referenceMeshInverseDensities.size() == _referenceRadii.size());
  Mesh points;
  // For each atom.
  for (std::size_t i = 0; i != _referenceMeshes.size(); ++i) {
    const Number radius = _referenceRadii[i];
    const Float area = 4. * numerical::Constants<Float>::Pi() *
                         radius * radius;
    const std::size_t numPoints =
      static_cast<std::size_t>(area * _pointDensity);
    // Record the inverse density.
    _referenceMeshInverseDensities[i] = area / numPoints;
    // Place points on a unit sphere.
    distributePointsOnSphereWithGoldenSectionSpiral<Point>
    (numPoints, std::back_inserter(points));
    // Scale by the radius.
    points *= radius;
    // Copy.
    _referenceMeshes[i] = points;
    points.clear();
  }
}


template<typename Float>
inline
void
Protein<Float>::
initializeAminoAcids(const std::vector<std::size_t>& aminoAcidSizes)
{
  assert(_aminoAcids.size() == aminoAcidSizes.size());
  std::size_t sum = 0;
  for (std::size_t i = 0; i != _aminoAcids.size(); ++i) {
    _aminoAcids[i].atoms = &_aminoAcidAtoms[0] + sum;
    _aminoAcids[i].size = aminoAcidSizes[i];
    sum += aminoAcidSizes[i];
  }
}


template<typename Float>
inline
void
Protein<Float>::
initializeRotamers(const std::vector<std::size_t>& numRotamersPerSite,
                   const std::vector<std::size_t>& rotamers)
{
  // Convert the amino acid indices to rotamers.
  std::vector<Rotamer> data(rotamers.size());
  const Point* c = &_rotamerAtomCenters[0];
  for (std::size_t i = 0; i != data.size(); ++i) {
    data[i].aminoAcid = &_aminoAcids[rotamers[i]];
    data[i].coordinates = c;
    c += _aminoAcids[rotamers[i]].size;
  }
  assert(std::size_t(c - &_rotamerAtomCenters[0]) ==
         _rotamerAtomCenters.size());
  _rotamers.rebuild(numRotamersPerSite.begin(), numRotamersPerSite.end(),
                    data.begin(), data.end());
}

template<typename Float>
inline
void
Protein<Float>::
makeRotamerBBoxes()
{
  // Build the array of arrays.
  {
    std::vector<std::size_t> sizes(_rotamers.getNumberOfArrays());
    for (std::size_t i = 0; i != sizes.size(); ++i) {
      sizes[i] = _rotamers.size(i);
    }
    _rotamerBBoxes.rebuild(sizes.begin(), sizes.end());
  }
  geom::Ball<Number, 3> ball;
  // Make a box around each rotamer.
  for (std::size_t i = 0; i != _rotamers.size(); ++i) {
    const Rotamer& r = _rotamers[i];
    BBox& box = _rotamerBBoxes[i];
    // Initialize the bounding box with the center of the first atom.
    box.lower = r.coordinates[0];
    box.upper = r.coordinates[0];
    // For each atom in the rotamer.
    for (std::size_t j = 0; j != r.aminoAcid->size; ++j) {
      ball.center = r.coordinates[j];
      ball.radius = _referenceRadii[r.aminoAcid->atoms[j].atomIndex];
      // Add the ball to the bounding box.
      box += ball;
    }
  }
  // Make a box for each site.
  assert(_siteBBoxes.size() == _rotamers.getNumberOfArrays());
  for (std::size_t i = 0; i != _siteBBoxes.size(); ++i) {
    // There must be at least one rotamer.
    assert(! _rotamers.empty(i));
    // Initialize with the first bounding box.
    _siteBBoxes[i] = _rotamerBBoxes(i, 0);
    // Add each of the other boxes.
    for (std::size_t j = 1; j != _rotamers.size(i); ++j) {
      _siteBBoxes[i] += _rotamerBBoxes(i, j);
    }
  }
}

template<typename Float>
inline
void
Protein<Float>::
calculateRotamerAreas(const std::vector<std::size_t>& numRotamersPerSite)
{
  // Allocate memory.
  _areaTriPeptideClipped.rebuild(numRotamersPerSite.begin(),
                                 numRotamersPerSite.end());
  _areaTemplateClipped.rebuild(numRotamersPerSite.begin(),
                               numRotamersPerSite.end());
  PhysicalAtom atom;
  std::vector<PhysicalAtom> rotamerPeptideAtoms, triPeptideClippingAtoms;
  std::vector<Mesh> rotamerPeptideMeshes;
  // For each site.
  for (std::size_t i = 0; i != _rotamers.getNumberOfArrays(); ++i) {
    // Add the peptide atoms.
    rotamerPeptideAtoms.clear();
    getPeptideAtoms(i, std::back_inserter(rotamerPeptideAtoms));
    // The atoms in the tri-peptide group which are not in the peptide
    // group clip the peptide and rotamer.
    triPeptideClippingAtoms.clear();
    for (std::size_t j = _triPeptideGroups[i]; j != _peptideGroups[i];
         ++j) {
      triPeptideClippingAtoms.push_back(_backbone[j]);
    }
    for (std::size_t j = _peptideGroups[i + 1]; j != _triPeptideGroups[i + 1];
         ++j) {
      triPeptideClippingAtoms.push_back(_backbone[j]);
    }
    const std::size_t numPeptide = _peptideGroups[i + 1] - _peptideGroups[i];
    // For each rotamer at the site.
    for (std::size_t j = 0; j != _rotamers.size(i); ++j) {
      // Erase the old rotamer.
      rotamerPeptideAtoms.erase(rotamerPeptideAtoms.begin() + numPeptide,
                                rotamerPeptideAtoms.end());
      // Add the atoms in the new rotamer.
      const Rotamer& r = _rotamers(i, j);
      const AminoAcid& aa = *r.aminoAcid;
      for (std::size_t k = 0; k != aa.size; ++k) {
        // Copy the LogicalAtom.
        atom = aa.atoms[k];
        // Add the center.
        atom.center = r.coordinates[k];
        rotamerPeptideAtoms.push_back(atom);
      }
      // Form the mesh of the surface of the peptide and rotamer atoms.
      meshBoundaryOfUnion(&rotamerPeptideMeshes, rotamerPeptideAtoms);
      // Clip with the tri-peptide backbone atoms.
      clipAndErase(&rotamerPeptideMeshes, rotamerPeptideAtoms,
                   triPeptideClippingAtoms);
      // Calculate the area.
      AreaTuple& area = _areaTriPeptideClipped(i, j);
      // Set both components to zero.
      std::fill(area.begin(), area.end(), 0.);
      // For each atom.
      for (std::size_t k = 0; k != rotamerPeptideAtoms.size(); ++k) {
        const PhysicalAtom& a = rotamerPeptideAtoms[k];
        // Accumulate the area.
        area[a.polarIndex] += rotamerPeptideMeshes[k].size() *
                              _referenceMeshInverseDensities[a.atomIndex];
      }
    }
  }
}

template<typename Float>
inline
void
Protein<Float>::
meshBoundaryOfUnion(std::vector<Mesh>* meshes,
                    const std::vector<PhysicalAtom>& atoms)
{
  // Initialize.
  meshes->resize(atoms.size());
  // For each atom in the set.
  for (std::size_t i = 0; i != atoms.size(); ++i) {
    // Start with the reference mesh.
    (*meshes)[i] = _referenceMeshes[atoms[i].atomIndex];
    // Reference for a more convenient name.
    Mesh& mesh = (*meshes)[i];
    // Translate to the specified center.
    mesh += atoms[i].center;
    // For each of the other spheres.
    for (std::size_t j = 0; j != atoms.size(); ++j) {
      if (i == j) {
        continue;
      }
      // If the two spheres intersect.
      if (doIntersect(atoms[i], atoms[j])) {
        // Clip the mesh and discard the clipped points.
        clipAndErase(&mesh, atoms[j]);
      }
    }
  }
}


template<typename Float>
inline
void
Protein<Float>::
clipAndErase(std::vector<Mesh>* meshes,
             const std::vector<PhysicalAtom>& meshedAtoms,
             const std::vector<PhysicalAtom>& clippingAtoms)
{
  assert(meshes->size() == meshedAtoms.size());
  // For each of the meshed atoms.
  for (std::size_t i = 0; i != meshes->size(); ++i) {
    // For each of the clipping atoms.
    for (std::size_t j = 0; j != clippingAtoms.size(); ++j) {
      // If the clipping atom intersects the meshed atom.
      if (doIntersect(meshedAtoms[i], clippingAtoms[j])) {
        // Clip the mesh and erase any clipped points.
        clipAndErase(&(*meshes)[i], clippingAtoms[j]);
      }
    }
  }
}

template<typename Float>
inline
void
Protein<Float>::
clipAndErase(Mesh* mesh, const PhysicalAtom& atom)
{
  // Clip the mesh.
  const std::size_t numActive =
    clip(mesh, mesh->size(), atom.center, _referenceRadii[atom.atomIndex]);
  // Discard the clipped points.
  mesh->erase(mesh->begin() + numActive, mesh->end());
}

} // namespace surfaceArea
}
