// -*- C++ -*-

#if !defined(__performance_PerformanceDataSerial_tcc__)
#error This file is an implementation detail of PerformanceDataSerial.
#endif


namespace stlib
{
namespace performance
{


inline
void
print(std::ostream& out, PerformanceData const& x)
{
  for (std::size_t i = 0; i != x.numericKeys.size(); ++i) {
    out << x.numericKeys[i] << " = " << x.numerics.at(x.numericKeys[i])
        << '\n';
  }
  for (std::size_t i = 0; i != x.timeKeys.size(); ++i) {
    out << x.timeKeys[i] << " = " << x.times.at(x.timeKeys[i]) << '\n';
  }
  out << "Total time = " << x.total.elapsed() << '\n';
}


inline
void
printCsv(std::ostream& out, PerformanceData const& x)
{
  _printCsv(out, x.numericKeys, x.numerics);
  _printCsv(out, x.timeKeys, x.times);
}


} // namespace performance
} // namespace stlib
