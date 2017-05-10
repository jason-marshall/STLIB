// -*- C++ -*-

/*!
  \file surfaceArea/Protein.h
  \brief A representation of a protein that is used for computing SASA.
*/

#if !defined(__surfaceArea_Protein_h__)
#define __surfaceArea_Protein_h__

#include "stlib/surfaceArea/Atom.h"
#include "stlib/surfaceArea/sphereMeshing.h"

#include "stlib/container/StaticArrayOfArrays.h"
#include "stlib/geom/kernel/Ball.h"
#include "stlib/geom/kernel/BBox.h"
#include "stlib/ext/array.h"
#include "stlib/ext/vector.h"

#include <utility>

namespace stlib
{
namespace surfaceArea
{

//! A representation of a protein that is used for computing SASA.
/*!
  \param Float The floating-point number type.

  Calculating the solvent-accessible surface areas (SASA) could logically be
  done with a function. However, the function signature would be complicated.
  Also the helper functions would need many parameters. Thus this functionality
  is implemented with a class. All of the work is done in the constructor,
  which takes all of the necessary input and calculates the SASA. Use the
  accessors to get the results.
*/
template<typename Float>
class Protein
{
  //
  // Public types.
  //
public:

  //! The floating-point number type.
  typedef Float Number;
  //! A Cartesian point.
  typedef std::array<Number, 3> Point;
  //! A tuple of polar/non-polar components of the area.
  typedef std::array<Number, 2> AreaTuple;
  //! A bounding box.
  typedef geom::BBox<Number, 3> BBox;
  //! A mesh is a set of points.
  typedef std::vector<Point> Mesh;
  //! A reference atom index, and a polar/non-polar index.
  typedef surfaceArea::LogicalAtom LogicalAtom;
  //! A reference atom index, a polar/non-polar index, and a location.
  typedef surfaceArea::PhysicalAtom<Number> PhysicalAtom;

  //
  // Nested classes.
  //
private:

  //! An amino acid is represented by a sequence of LogicalAtom's.
  struct AminoAcid {
    //! Const pointer to the beginning of the atoms.
    const LogicalAtom* atoms;
    //! The number of atoms.
    std::size_t size;
  };

  //! A rotamer is an amino acid along with the atom coordinates.
  struct Rotamer {
    //! The amino acid.
    const AminoAcid* aminoAcid;
    //! The atom center coordinates.
    const Point* coordinates;
  };

  //
  // Member data.
  //
private:

  // The reference atoms give properties for each type of atom.

  //! The radius of each type of atom, expanded by the probe radius.
  std::vector<Number> _referenceRadii;
  //! A set of points distributed on the surface of each (expanded) reference atom.
  std::vector<Mesh> _referenceMeshes;
  //! The inverse point densities (area in square Angstroms per point) for the reference atom meshes.
  std::vector<Number> _referenceMeshInverseDensities;

  // Backbone.

  //! The atoms in the backbone.
  /*! All of the atoms that are in peptide groups must be contiguous in
    memory. The peptide groups and tri-peptide groups are defined by
    delimiters. The end of one group is the beginning of the next group. */
  std::vector<PhysicalAtom> _backbone;
  //! The delimiters that define the peptide groups in the backbone.
  std::vector<std::size_t> _peptideGroups;
  //! The delimiters that define the tri-peptide groups in the backbone.
  std::vector<std::size_t> _triPeptideGroups;

  // Rotamers.

  //! The amino acids.
  std::vector<AminoAcid> _aminoAcids;
  //! The atoms (type index and polar/non-polar index) that comprise the amino acids.
  std::vector<LogicalAtom> _aminoAcidAtoms;

  //! The rotamers, grouped by site.
  container::StaticArrayOfArrays<Rotamer> _rotamers;
  //! The positions of the atoms in the rotamers.
  std::vector<Point> _rotamerAtomCenters;

  // Rotamer bounding boxes.

  //! A bounding box around each rotamer conformation.
  container::StaticArrayOfArrays<BBox> _rotamerBBoxes;
  //! A bounding box around all rotomers at each site.
  std::vector<BBox> _siteBBoxes;

  // Areas.

  //! The area of the peptide and rotamer, clipped by the local tri-peptide backbone.
  container::StaticArrayOfArrays<AreaTuple> _areaTriPeptideClipped;
  //! The area of the peptide and rotamer, clipped by the full template.
  container::StaticArrayOfArrays<AreaTuple> _areaTemplateClipped;

  // Parameters.

  //! The probe radius. The atomic radii are increased by this amount in generating the SAS.
  Number _probeRadius;
  //! The target density (points per square Angstrom) for placing points on the spheres.
  Number _pointDensity;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Calculate the SASA.
  /*!
    \param referenceAtomRadii The radius of each type of atom.

    \param backbone The atoms in the backbone.
    \param peptideGroups The delimiters that define the peptide groups.
    \param triPeptideGroups The delimiters that define the tri-peptide groups.

    \param aminoAcidSizes The number of atoms in each amino acid.
    \param aminoAcidAtoms The atoms (type index and polar/non-polar
    index) that comprise the amino acids.
    \param numRotamersPerSite The number of rotamers used at each site.
    \param rotamers Sequence of amino acid indices.
    \param rotamerAtomCenters The positions of the atoms in the rotamers.

    \param probeRadius The probe radius. The default value is 1.4 Angstroms.
    \param pointDensity The target density (points per square Angstrom)
    for placing points on the spheres. The default value is 10.
  */
  Protein(// Reference atoms.
    const std::vector<Number>& referenceAtomRadii,
    // Backbone.
    const std::vector<PhysicalAtom>& backbone,
    const std::vector<std::size_t>& peptideGroups,
    const std::vector<std::size_t>& triPeptideGroups,
    // Rotamers.
    const std::vector<std::size_t>& aminoAcidSizes,
    const std::vector<LogicalAtom>& aminoAcidAtoms,
    const std::vector<std::size_t>& numRotamersPerSite,
    const std::vector<std::size_t>& rotamers,
    const std::vector<Point>& rotamerAtomCenters,
    // Parameters.
    const Number probeRadius = 1.4,
    const Number pointDensity = 10) :
    // Reference atoms.
    _referenceRadii(referenceAtomRadii),
    _referenceMeshes(referenceAtomRadii.size()),
    _referenceMeshInverseDensities(referenceAtomRadii.size()),
    // Backbone.
    _backbone(backbone),
    _peptideGroups(peptideGroups),
    _triPeptideGroups(triPeptideGroups),
    // Rotamers.
    _aminoAcids(aminoAcidSizes.size()),
    _aminoAcidAtoms(aminoAcidAtoms),
    _rotamers(),
    _rotamerAtomCenters(rotamerAtomCenters),
    // Rotamer bounding boxes.
    _rotamerBBoxes(),
    _siteBBoxes(numRotamersPerSite.size()),
    // Areas.
    _areaTriPeptideClipped(),
    _areaTemplateClipped(),
    // Parameters.
    _probeRadius(probeRadius),
    _pointDensity(pointDensity)
  {
    assert(peptideGroups.size() == triPeptideGroups.size());
    assert(ext::sum(aminoAcidSizes) == aminoAcidAtoms.size());
    assert(numRotamersPerSite.size() + 1 == peptideGroups.size());
    assert(ext::sum(numRotamersPerSite) == rotamers.size());
    // Check the amino acid indices.
    for (std::size_t i = 0; i != rotamers.size(); ++i) {
      assert(rotamers[i] < aminoAcidSizes.size());
    }
    assert(probeRadius > 0);
    assert(pointDensity > 0);

    // Expand the atomic radii by the probe radius.
    _referenceRadii += _probeRadius;
    // Generate meshes for each of the reference atoms.
    meshReferenceAtoms();
    // Initialize the amino acids.
    initializeAminoAcids(aminoAcidSizes);
    // Initialize the rotamers.
    initializeRotamers(numRotamersPerSite, rotamers);
    // Calculate the rotamer bounding boxes.
    makeRotamerBBoxes();
    // Calculate the rotamer areas.
    calculateRotamerAreas(numRotamersPerSite);
  }

  // Use the default destructor.

private:

  // Since we defined a constructor there is no default constructor.
  // The copy constructor is not implemented.
  Protein(const Protein& other);
  // The assignment operator is not implemented.
  Protein&
  operator=(const Protein& other);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return a vector of the meshes for the (expanded) reference atoms.
  const std::vector<Mesh>&
  getReferenceMeshes() const
  {
    return _referenceMeshes;
  }

  //! Return the probe radius.
  Number
  getProbeRadius() const
  {
    return _probeRadius;
  }

  //! Return the target density (points per square Angstrom) for placing points on the spheres.
  Number
  getPointDensity() const
  {
    return _pointDensity;
  }

  //! Return the area of the specified rotamer at the specified site.
  /*! The area includes the area of the local template and is clipped by
    the local tri-peptide atoms. */
  const AreaTuple&
  getRotamerAreaTriPeptideClipped(const std::size_t site,
                                  const std::size_t rotamer) const
  {
    return _areaTriPeptideClipped(site, rotamer);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Functions.
  //@{
private:

  //! Generate meshes for each of the reference atoms.
  void
  meshReferenceAtoms();

  //! Initialize the amino acids.
  void
  initializeAminoAcids(const std::vector<std::size_t>& aminoAcidSizes);

  //! Initialize the rotamers.
  void
  initializeRotamers(const std::vector<std::size_t>& numRotamersPerSite,
                     const std::vector<std::size_t>& rotamers);

  //! Calculate the rotamer bounding boxes.
  void
  makeRotamerBBoxes();

  //! Calculate the rotamer areas.
  void
  calculateRotamerAreas(const std::vector<std::size_t>& numRotamersPerSite);

  //! Generate a set of points that represents the surface of the union of atoms.
  /*! The points are generated on a per-atom basis, that is the output is a
    list of meshes, one for each atom. */
  void
  meshBoundaryOfUnion(std::vector<Mesh>* meshes,
                      const std::vector<PhysicalAtom>& atoms);

  //! Clip the meshes with the specified atoms. Erase the clipped points.
  void
  clipAndErase(std::vector<Mesh>* meshes,
               const std::vector<PhysicalAtom>& meshedAtoms,
               const std::vector<PhysicalAtom>& clippingAtoms);

  //! Clip the mesh with the specified atom. Erase the clipped points.
  void
  clipAndErase(Mesh* mesh, const PhysicalAtom& atom);

  //! Return true if the two atoms intersect.
  bool
  doIntersect(const PhysicalAtom& a, const PhysicalAtom& b) const
  {
    return ext::squaredDistance(a.center, b.center) <
           (_referenceRadii[a.atomIndex] + _referenceRadii[b.atomIndex]) *
           (_referenceRadii[a.atomIndex] + _referenceRadii[b.atomIndex]);
  }

  //! Get the peptide atoms for the specified site.
  template<typename _OutputIterator>
  void
  getPeptideAtoms(const std::size_t site, _OutputIterator atoms) const
  {
    for (std::size_t i = _peptideGroups[site]; i != _peptideGroups[site + 1];
         ++i) {
      *atoms++ = _backbone[i];
    }
  }

  //@}
};

} // namespace surfaceArea
}

#define __surfaceArea_Protein_ipp__
#include "stlib/surfaceArea/Protein.ipp"
#undef __surfaceArea_Protein_ipp__

#endif
