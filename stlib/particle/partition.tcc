// -*- C++ -*-

#if !defined(__particle_partition_tcc__)
#error This file is an implementation detail of partition.
#endif

namespace stlib
{
namespace particle
{


// Merge two data tables.
// The value type _T must have an associated operator+=().
template<typename _T>
inline
void
merge(const std::vector<std::pair<IntegerTypes::Code, _T> >& a,
      const std::vector<std::pair<IntegerTypes::Code, _T> >& b,
      std::vector<std::pair<IntegerTypes::Code, _T> >* merged)
{
  merged->clear();
  // Indices for the two inputs.
  std::size_t i = 0;
  std::size_t j = 0;
  // Add one element to the merged result so that we don't need to check if
  // it is empty.
  if (! a.empty()) {
    if (! b.empty()) {
      if (a[0].first <= b[0].first) {
        merged->push_back(a[0]);
        ++i;
      }
      else {
        merged->push_back(b[0]);
        ++j;
      }
    }
    else {
      merged->push_back(a[0]);
      ++i;
    }
  }
  else {
    if (! b.empty()) {
      merged->push_back(b[0]);
      ++j;
    }
  }
  // Loop until one or the other inputs are exhausted.
  while (i != a.size() && j != b.size()) {
    if (a[i].first <= b[j].first) {
      if (! merged->empty() && merged->back().first == a[i].first) {
        merged->back().second += a[i].second;
      }
      else {
        merged->push_back(a[i]);
      }
      ++i;
    }
    else {
      if (! merged->empty() && merged->back().first == b[j].first) {
        merged->back().second += b[j].second;
      }
      else {
        merged->push_back(b[j]);
      }
      ++j;
    }
  }
  // Finish off a.
  for (; i != a.size(); ++i) {
    if (! merged->empty() && merged->back().first == a[i].first) {
      merged->back().second += a[i].second;
    }
    else {
      merged->push_back(a[i]);
    }
  }
  // Finish off b.
  for (; j != b.size(); ++j) {
    if (! merged->empty() && merged->back().first == b[j].first) {
      merged->back().second += b[j].second;
    }
    else {
      merged->push_back(b[j]);
    }
  }
}


// Merge the second table into the first.
// The value type _T must have an associated operator+=().
template<typename _T>
inline
void
merge(std::vector<std::pair<IntegerTypes::Code, _T> >* first,
      const std::vector<std::pair<IntegerTypes::Code, _T> >& second)
{
  std::vector<std::pair<IntegerTypes::Code, _T> > merged;
  merge(*first, second, &merged);
  first->swap(merged);
}


// Shift the codes by the specified number of levels. Then compress to merge
// duplicate codes.
template<std::size_t _Dimension, typename _T>
inline
void
shift(std::vector<std::pair<IntegerTypes::Code, _T> >* table,
      std::size_t numLevels)
{
  typedef IntegerTypes::Code Code;

  assert(numLevels > 0);
  // Dispense with the trivial case.
  if (table->empty()) {
    return;
  }
  // Shift the codes.
  for (std::size_t i = 0; i != table->size(); ++i) {
    (*table)[i].first >>= _Dimension * numLevels;
  }
  // Merge duplicates.
  std::vector<std::pair<Code, _T> > tmp;
  // Add the first one.
  tmp.push_back((*table)[0]);
  // Add the rest.
  for (std::size_t i = 1; i != table->size(); ++i) {
    if ((*table)[i].first == tmp.back().first) {
      tmp.back().second += (*table)[i].second;
    }
    else {
      tmp.push_back((*table)[i]);
    }
  }
  table->swap(tmp);
}


} // namespace particle
}
