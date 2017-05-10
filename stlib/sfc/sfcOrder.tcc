// -*- C++ -*-

#if !defined(stlib_sfc_sfcOrder_tcc)
#error This file is an implementation detail of sfcOrder.
#endif

namespace stlib
{
namespace sfc
{


template<typename Index, typename Code, typename Float, std::size_t D>
inline
std::vector<Index>
sfcOrderSpecific(std::vector<std::array<Float, D> > const& locations)
{
  using stlib::performance::start;
  using stlib::performance::stop;

  stlib::performance::Scope _("stlib::sfc::sfcOrderSpecific()");

  // First check the trivial case.
  if (locations.empty()) {
    return std::vector<Index>{};
  }

  // Put a bounding box around the locations. Then use the bounding box
  // to define the SFC grid.
  start("Build the SFC grid");
  LocationCode<Traits<D, Float, Code> > const
    grid(geom::bbox(locations.begin(), locations.end()));
  stop();

  // Generate codes for the locations.
  start("Calculate SFC codes");
  std::vector<std::pair<Code, std::size_t> > codeIndexPairs(locations.size());
  for (std::size_t i = 0; i != codeIndexPairs.size(); ++i) {
    codeIndexPairs[i].first = grid.code(locations[i]);
    codeIndexPairs[i].second = i;
  }
  stop();

  start("Sort by the codes");
  lorg::sort(&codeIndexPairs, grid.numBits());
  stop();

  start("Extract the vector of indices");
  std::vector<Index> result(codeIndexPairs.size());
  for (std::size_t i = 0; i != result.size(); ++i) {
    result[i] = codeIndexPairs[i].second;
  }
  stop();

  return result;
}


} // namespace sfc
} // namespace stlib
