// -*- C++ -*-

#if !defined(__performance_PerformanceSerial_tcc__)
#error This file is an implementation detail of PerformanceSerial.
#endif


namespace stlib
{
namespace performance
{


#ifdef STLIB_PERFORMANCE
inline
void
print(std::ostream& out)
{
  Performance const& x = getInstance();
  for (std::size_t i = 0; i != x.scopeKeys.size(); ++i) {
    PerformanceData const& data = x.scopes.at(x.scopeKeys[i]);
    if (! data.empty()) {
      out << '\n' << x.scopeKeys[i] << ":\n";
      print(out, data);
    }
  }
}
#else
inline
void
print(std::ostream&)
{
}
#endif


#ifdef STLIB_PERFORMANCE
inline
void
printCsv(std::ostream& out)
{
  Performance const& x = getInstance();
  for (std::size_t i = 0; i != x.scopeKeys.size(); ++i) {
    PerformanceData const& data = x.scopes.at(x.scopeKeys[i]);
    if (! data.empty()) {
      out << '\n' << x.scopeKeys[i] << ":\n";
      printCsv(out, data);
    }
  }
}
#else
inline
void
printCsv(std::ostream&)
{
}
#endif


} // namespace performance
} // namespace stlib
