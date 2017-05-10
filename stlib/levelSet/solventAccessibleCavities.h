// -*- C++ -*-

#if !defined(__levelSet_solventAccessibleCavities_h__)
#define __levelSet_solventAccessibleCavities_h__

#include "stlib/levelSet/solventExcluded.h"
#include "stlib/levelSet/marchingSimplices.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetSolventAccessibleCavities Solvent-Accessible Cavities
These functions calculate level sets related to the solvent-accessible cavities.
*/
//@{


//! Compute the seed balls that define the SAC.
/*!
  \param balls The atoms that comprise the molecule.
  \param probeRadius The radius of the solvent probe.
  \param targetGridSpacing The target spacing for the AMR grid.
  \param seeds The seed balls will be appended to this vector.
*/
template<typename _T, std::size_t _D>
void
solventAccessibleCavitySeeds(std::vector<geom::Ball<_T, _D> > balls,
                             _T probeRadius, _T targetGridSpacing,
                             std::vector<geom::Ball<_T, _D> >* seeds);

//! Compute the volume and surface area of each component of the SAC.
/*!
  \param balls The atoms that comprise the molecule.
  \param probeRadius The radius of the solvent probe.
  \param targetGridSpacing The target spacing for the AMR grid.
  \param content The vector of volumes for each component of the SAC.
  \param boundary The vector of surface areas for each component of the SAC.
*/
template<typename _T, std::size_t _D>
void
solventAccessibleCavities(const std::vector<geom::Ball<_T, _D> >& balls,
                          _T probeRadius, _T targetGridSpacing,
                          std::vector<_T>* content,
                          std::vector<_T>* boundary);

//! Construct a level set for the solvent-accessible surface for solvents inside the protein.
/*! Note that we pass the vector of balls by value as we will manipulate the
  radii internally. */
template<typename _T, std::size_t _D, std::size_t N>
void
solventAccessibleCavities(Grid<_T, _D, N>* grid,
                          std::vector<geom::Ball<_T, _D> > balls,
                          _T probeRadius);

//! Construct a level set for the solvent-accessible surface for solvents inside the protein.
template<typename _T, std::size_t _D>
void
solventAccessibleCavities(container::SimpleMultiArrayRef<_T, _D>* grid,
                          const geom::BBox<_T, _D>& domain,
                          const std::vector<geom::Ball<_T, _D> >& balls,
                          _T probeRadius);


//! Construct a level set for the solvent-accessible surface for solvents inside the protein.
template<typename _T, std::size_t _D>
inline
void
solventAccessibleCavities(GridUniform<_T, _D>* grid,
                          const std::vector<geom::Ball<_T, _D> >& balls,
                          const _T probeRadius)
{
  solventAccessibleCavities(grid, grid->domain(), balls, probeRadius);
}


//@}

} // namespace levelSet
}

#define __levelSet_solventAccessibleCavities_ipp__
#include "stlib/levelSet/solventAccessibleCavities.ipp"
#undef __levelSet_solventAccessibleCavities_ipp__

#endif
