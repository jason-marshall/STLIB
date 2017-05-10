// -*- C++ -*-

#if !defined(__performance_PerformanceMpi_tcc__)
#error This file is an implementation detail of PerformanceMpi.
#endif


namespace stlib
{
namespace performance
{


#ifdef STLIB_PERFORMANCE
inline
void
print(std::ostream& out, MPI_Comm const comm)
{
  Performance const& x = getInstance();
  for (std::size_t i = 0; i != x.scopeKeys.size(); ++i) {
    PerformanceData const& data = x.scopes.at(x.scopeKeys[i]);
    if (mpi::commRank(comm) == 0) {
      out << '\n' << x.scopeKeys[i] << ":\n";
    }
    print(out, data, comm);
  }
}
#else
inline
void
print(std::ostream&, MPI_Comm)
{
}
#endif


#ifdef STLIB_PERFORMANCE
inline
void
printCsv(std::ostream& out, MPI_Comm const comm)
{
  Performance const& x = getInstance();
  for (std::size_t i = 0; i != x.scopeKeys.size(); ++i) {
    PerformanceData const& data = x.scopes.at(x.scopeKeys[i]);
    if (mpi::commRank(comm) == 0) {
      out << '\n' << x.scopeKeys[i] << ":\n";
    }
    printCsv(out, data, comm);
  }
}
#else
inline
void
printCsv(std::ostream&, MPI_Comm)
{
}
#endif


} // namespace performance
} // namespace stlib
