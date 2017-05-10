// -*- C++ -*-

#if !defined(__elc_EulerianCommShell_ipp__)
#error This file is an implementation detail of the class EulerianCommShell.
#endif

namespace stlib
{
namespace elc
{


template<std::size_t N, typename T>
inline
void
EulerianCommShell<N, T>::
sendPressure()
{
  //
  // Send the pressures.
  //
  const std::size_t numComm = _lagrangianData.size();
  _pressureRequests.resize(numComm);
  int faceIndex = 0;
  // For each Lagrangian processor with which we are communicating.
  for (std::size_t i = 0; i != numComm; ++i) {
    // We free these here to match the boundary case.
    _identifiers[i].resize(0);
    // The Lagrangian processor rank in the world communicator.
    const int lagrangianProcessor = _lagrangianData[i][0];
    // The pressures are defined at the faces.  Thus the number of
    // pressures is the same as the number of faces.
    assert(_pressures[i].size() == _lagrangianData[i][2]);
    // Copy the pressures from the assembled mesh.
    for (std::size_t j = 0; j != _pressures[i].size(); ++j) {
      // Extract the pressure.
      _pressures[i][j] = _assembledPressures[faceIndex];
      ++faceIndex;
    }
#ifdef ELC_USE_CPP_INTERFACE
    _pressureRequests[i] =
      _comm.Isend(&_pressures[i][0], _pressures[i].size(),
                  _mpiNumber, lagrangianProcessor, TagPressures);
#else
    MPI_Isend(&_pressures[i][0], _pressures[i].size(),
              _mpiNumber, lagrangianProcessor, TagPressures, _comm,
              &_pressureRequests[i]);
#endif
  }
}



template<std::size_t N, typename T>
inline
void
EulerianCommShell<N, T>::
waitForPressure()
{
  const std::size_t numComm = _lagrangianData.size();
  // For each Lagrangian processor with which we are communicating.
  for (std::size_t i = 0; i != numComm; ++i) {
#ifdef ELC_USE_CPP_INTERFACE
    _pressureRequests[i].Wait();
#else
    MPI_Wait(&_pressureRequests[i], MPI_STATUS_IGNORE);
#endif
    // Free the pressure memory for the i_th processor.
    _pressures[i].resize(0);
  }
}



template<std::size_t N, typename T>
inline
void
EulerianCommShell<N, T>::
initializePressure()
{
  const std::size_t numComm = _lagrangianData.size();
  _pressures.resize(numComm);
  for (std::size_t i = 0; i != numComm; ++i) {
    const int numFacesToReceive = _lagrangianData[i][2];
    _pressures[i].resize(numFacesToReceive);
  }

  // Allocate memory.
  _assembledPressures.resize(_assembledConnectivities.size());

  // Fill the pressures with a flag value.
  std::fill(_assembledPressures.begin(), _assembledPressures.end(),
            std::numeric_limits<Number>::max());
}

} // namespace elc
}
