// -*- C++ -*-

#include "stlib/sfc/AdaptiveCells.h"

int
main()
{
  constexpr std::size_t Dimension = 3;
  std::size_t const DefaultMaxObjectsPerCell =
    stlib::sfc::AdaptiveCells<stlib::sfc::Traits<Dimension>, void, false>::
    DefaultMaxObjectsPerCell;
  std::cout << "numGlobalObjects, commSize, maxObjectsPerCell, cellsSerialLowerBound, cellsDistributedLowerBound\n";
  for (int i = -1; i != 30; ++i) {
    std::size_t const numGlobalObjects = std::size_t(1) << i;
    for (int j = 0; j != 14; ++j) {
      int const commSize = 1 << j;
      std::size_t const maxObjectsPerCell =
        stlib::sfc::maxObjectsPerCellDistributed<Dimension>
        (numGlobalObjects, commSize);
      assert(maxObjectsPerCell > 0);
      std::cout << numGlobalObjects << ", " << commSize << ", "
                << maxObjectsPerCell << ", "
                << numGlobalObjects / commSize / DefaultMaxObjectsPerCell << ", "
                << numGlobalObjects / maxObjectsPerCell << '\n';
    }
  }
}
