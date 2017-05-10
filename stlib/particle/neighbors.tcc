// -*- C++ -*-

#if !defined(__particle_neighbors_tcc__)
#error This file is an implementation detail of neighbors.
#endif

namespace stlib
{
namespace particle
{


inline
NeighborsPerformance::
NeighborsPerformance() :
  _potentialNeighborsCount(),
  _neighborsCount(),
  _timer(),
  _timePotentialNeighbors(0),
  _timeNeighbors(0)
{
}


inline
void
NeighborsPerformance::
printPerformanceInfo(std::ostream& out) const
{
  out << "Find potential neighbors count = " << _potentialNeighborsCount
      << '\n';
  // Time totals.
  {
    const std::size_t Num = 2;
    const std::array<const char*, Num> names = {{
        "PotentialNeighbors",
        "Neighbors"
      }
    };
    const std::array<double, Num> values = {
      {
        _timePotentialNeighbors,
        _timeNeighbors
      }
    };
    out << "\nTime totals:\n";
    // Column headers.
    for (std::size_t i = 0; i != names.size(); ++i) {
      out << names[i];
      if (i != names.size() - 1) {
        out << ',';
      }
      else {
        out << '\n';
      }
    }
    // Values.
    for (std::size_t i = 0; i != values.size(); ++i) {
      out << values[i];
      if (i != values.size() - 1) {
        out << ',';
      }
      else {
        out << '\n';
      }
    }
  }
  // Time per operation.
  {
    const std::size_t Num = 2;
    const std::array<const char*, Num> names = {{
        "PotentialNeighbors",
        "Neighbors"
      }
    };
    const std::array<double, Num> values = {
      {
        _timePotentialNeighbors / _potentialNeighborsCount,
        _timeNeighbors / _neighborsCount
      }
    };
    out << "\nTime per operation:\n";
    // Column headers.
    for (std::size_t i = 0; i != names.size(); ++i) {
      out << names[i];
      if (i != names.size() - 1) {
        out << ',';
      }
      else {
        out << '\n';
      }
    }
    // Values.
    for (std::size_t i = 0; i != values.size(); ++i) {
      out << values[i];
      if (i != values.size() - 1) {
        out << ',';
      }
      else {
        out << '\n';
      }
    }
  }
}


} // namespace particle
}
