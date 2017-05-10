// -*- C++ -*-

#if !defined(__performance_PerformanceData_tcc__)
#error This file is an implementation detail of PerformanceData.
#endif


namespace stlib
{
namespace performance
{


inline
bool
PerformanceData::
empty() const BOOST_NOEXCEPT
{
  return numerics.empty() && times.empty();
}


inline
void
PerformanceData::
record(std::string const& key, double const value)
{
  if (numerics.count(key)) {
    numerics[key] += value;
  }
  else {
    numericKeys.push_back(key);
    numerics[key] = value;
  }
}


inline
void
PerformanceData::
start(std::string const& key)
{
  auto const iter = times.find(key);
  if (iter == times.end()) {
    timeKeys.push_back(key);
    _current = &times[key];
    *_current = 0;
  }
  else {
    _current = &iter->second;
  }    
  _timer.start();
}


inline
void
PerformanceData::
stop()
{
  _timer.stop();
  if (! _current) {
    throw std::runtime_error("PerformaceData::stop() called without a matching start().");
  }
  *_current += _timer.elapsed();
  _current = 0;
}


template<typename _T>
inline
void
_printCsvRow(std::ostream& out, std::vector<_T> const& values) BOOST_NOEXCEPT
{
  if (! values.empty()) {
    out << values[0];
    for (std::size_t i = 1; i != values.size(); ++i) {
      out << ',' << values[i];
    }
    out << '\n';
  }
}


inline
void
_printCsv(std::ostream& out,
          std::vector<std::string> const& keys,
          std::map<std::string, double> const& data)
{
  _printCsvRow(out, keys);
  std::vector<double> row;
  for (std::size_t i = 0; i != keys.size(); ++i) {
    row.push_back(data.at(keys[i]));
  }
  _printCsvRow(out, row);
}


} // namespace performance
} // namespace stlib
