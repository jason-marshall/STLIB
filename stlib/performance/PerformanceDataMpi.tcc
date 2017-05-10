// -*- C++ -*-

#if !defined(__performance_PerformanceDataMpi_tcc__)
#error This file is an implementation detail of PerformanceDataMpi.
#endif


namespace stlib
{
namespace performance
{


inline
void
_printCsv(std::ostream& out, MPI_Comm comm,
          std::vector<std::string> const& keys,
          std::map<std::string, double> const& data)
{
  int const rank = mpi::commRank(comm);
  if (rank == 0) {
    _printCsvRow(out, keys);
  }
  std::vector<double> row;
  for (std::size_t i = 0; i != keys.size(); ++i) {
    row.push_back(mpi::reduce(data.at(keys[i]), MPI_MAX, comm));
  }
  if (rank == 0) {
    _printCsvRow(out, row);
  }
}


inline
void
print(std::ostream& out, PerformanceData const& x, MPI_Comm const comm)
{
  for (std::size_t i = 0; i != x.numericKeys.size(); ++i) {
    mpi::printStatistics(out, x.numericKeys[i],
                         x.numerics.at(x.numericKeys[i]), comm);
  }
  for (std::size_t i = 0; i != x.timeKeys.size(); ++i) {
    mpi::printStatistics(out, x.timeKeys[i], x.times.at(x.timeKeys[i]),
                         comm);
  }
  mpi::printStatistics(out, "Total time", x.total.elapsed(), comm);
}


inline
void
printCsv(std::ostream& out, PerformanceData const& x, MPI_Comm const comm)
{
  _printCsv(out, comm, x.numericKeys, x.numerics);
  _printCsv(out, comm, x.timeKeys, x.times);
}


} // namespace performance
}
