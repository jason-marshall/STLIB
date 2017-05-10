// -*- C++ -*-

#include "stlib/surfaceArea/Protein.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef surfaceArea::Protein<double> Protein;
  typedef Protein::Number Number;
  typedef Protein::Point Point;
  typedef Protein::Mesh Mesh;
  typedef surfaceArea::PhysicalAtom<Number> PhysicalAtom;
  typedef surfaceArea::LogicalAtom LogicalAtom;

  // Two reference atoms. No backbone. No rotamers.
  {
    // Reference atoms.
    std::vector<Number> referenceAtomRadii;
    referenceAtomRadii.push_back(2.);
    referenceAtomRadii.push_back(3.);
    // Backbone.
    std::vector<PhysicalAtom> backbone;
    std::vector<std::size_t> peptideGroups(1);
    std::vector<std::size_t> triPeptideGroups(1);
    // Rotamers.
    std::vector<std::size_t> aminoAcidSizes;
    std::vector<LogicalAtom> aminoAcidAtoms;
    std::vector<std::size_t> numRotamersPerSite;
    std::vector<std::size_t> rotamers;
    std::vector<Point> rotamerAtomCenters;


    Protein x(referenceAtomRadii, backbone, peptideGroups,
              triPeptideGroups, aminoAcidSizes,
              aminoAcidAtoms, numRotamersPerSite, rotamers,
              rotamerAtomCenters);
    // The correct number of meshes.
    assert(x.getReferenceMeshes().size() == referenceAtomRadii.size());
    // For each mesh.
    for (std::size_t i = 0; i != x.getReferenceMeshes().size(); ++i) {
      const Number radius = referenceAtomRadii[i] + x.getProbeRadius();
      const Number area = 4. * numerical::Constants<Number>::Pi() *
                          radius * radius;
      const Mesh& mesh = x.getReferenceMeshes()[i];
      // Check the density of points.
      assert(std::abs(mesh.size() / area - x.getPointDensity()) <
             2. / area);
      for (std::size_t j = 0; j != mesh.size(); ++j) {
        // Check that the points have the correct radius.
        assert(std::abs(ext::magnitude(mesh[j]) - radius) < 10. *
               std::numeric_limits<Number>::epsilon());
      }
    }
  }

  {
    //
    // Reference atoms.
    //
    std::vector<Number> referenceAtomRadii;
    // Hydrogen
    referenceAtomRadii.push_back(1.2);
    // Carbon
    referenceAtomRadii.push_back(1.7);
    // Nitrogen
    referenceAtomRadii.push_back(1.55);
    // Oxygen
    referenceAtomRadii.push_back(1.52);

    //
    // Backbone.
    //
    // [HCA, C, O], [N, HN, CA, HCA, C, O], [N, HN, CA]
    std::vector<PhysicalAtom> backbone;
    PhysicalAtom atom;
    atom.atomIndex = 0;
    atom.polarIndex = 0;
    atom.center = Point{{0., 0., 0.}};
    // HCA
    atom.atomIndex = 0;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // C
    atom.atomIndex = 1;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // O
    atom.atomIndex = 3;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // N
    atom.atomIndex = 2;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // HN
    atom.atomIndex = 0;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // CA
    atom.atomIndex = 1;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // HCA
    atom.atomIndex = 0;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // C
    atom.atomIndex = 1;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // O
    atom.atomIndex = 3;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // N 2
    atom.atomIndex = 2;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // HN 0
    atom.atomIndex = 0;
    backbone.push_back(atom);
    atom.center[0] += 2;
    // CA 1
    atom.atomIndex = 1;
    backbone.push_back(atom);
    atom.center[0] += 2;

    // [HCA, C, O], [N, HN, CA, HCA, C, O], [N, HN, CA]
    std::vector<std::size_t> peptideGroups;
    peptideGroups.push_back(3);
    peptideGroups.push_back(9);
    std::vector<std::size_t> triPeptideGroups;
    triPeptideGroups.push_back(0);
    triPeptideGroups.push_back(12);

    //
    // Rotamers.
    //
    // One reference rotamer.
    std::vector<LogicalAtom> aminoAcidAtoms;
    {
      // CH3
      LogicalAtom a = {0, 0};
      // C
      a.atomIndex = 1;
      aminoAcidAtoms.push_back(a);
      // H
      a.atomIndex = 0;
      aminoAcidAtoms.push_back(a);
      aminoAcidAtoms.push_back(a);
      aminoAcidAtoms.push_back(a);
    }
    std::vector<std::size_t> aminoAcidSizes(1);
    aminoAcidSizes[0] = aminoAcidAtoms.size();

    // One site, one rotamer.
    std::vector<std::size_t> numRotamersPerSite(1);
    numRotamersPerSite[0] = 1;
    std::vector<std::size_t> rotamers(1);
    rotamers[0] = 0;
    std::vector<Point> rotamerAtomCenters(4);
    rotamerAtomCenters[0] = Point{{10, 2, 0}};
    rotamerAtomCenters[1] = Point{{12, 2, 0}};
    rotamerAtomCenters[2] = Point{{10, 2, 2}};
    rotamerAtomCenters[3] = Point{{8, 2, 0}};

    Protein x(referenceAtomRadii, backbone, peptideGroups,
              triPeptideGroups, aminoAcidSizes,
              aminoAcidAtoms, numRotamersPerSite, rotamers,
              rotamerAtomCenters);
  }

  return 0;
}
