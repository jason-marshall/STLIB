// -*- C++ -*-

#if !defined(__particle_lookup_tcc__)
#error This file is an implementation detail of lookup.
#endif

namespace stlib
{
namespace particle
{

inline
void
LookupTable::
initialize(const std::vector<Code>& codes, const std::size_t maxSize)
{
  // There must, at least, be a guard element.
  assert(! codes.empty());
  assert(maxSize >= 2);
  // Handle the trivial case.
  if (codes.size() == 1) {
    _table.resize(1);
    _table[0] = 0;
    _shift = 0;
    _first = 0;
    return;
  }

  // Determine an appropriate shift given the maximum size.
  _shift = 0;
  Code front = codes.front();
  Code back = codes[codes.size() - 2];
  // Note that we have one element in the table that is one past the last code
  // and whose value is the number of valid codes.
  while (back + 1 - front + 1 > maxSize) {
    ++_shift;
    front >>= 1;
    back >>= 1;
  }
  // Resize the table and set the first (shifted) code.
  _table.resize(back + 1 - front + 1);
  _first = front;
  // Check that the guard element is valid.
  assert((codes.back() >> _shift) - _first >= _table.size() - 1);

  // Set the values in the table.
  std::size_t n = 0;
  for (std::size_t i = 0; i != _table.size(); ++i) {
    while ((codes[n] >> _shift) - _first < i) {
      ++n;
    }
    _table[i] = n;
  }
}


inline
LookupTable::result_type
LookupTable::
operator()(argument_type code) const
{
#ifdef STLIB_DEBUG
  assert(! _table.empty());
#endif
  // First shift the code and see if it lies before the beginning of the table.
  code >>= _shift;
  if (code < _first) {
    return 0;
  }
  // Then offset the code and see if it lies after the end of the table.
  code -= _first;
  if (code >= _table.size()) {
    // The last element of the table is the size of the array of codes.
    // This value indicates that there are no codes greater than or
    // equal to the argument.
    return _table.back();
  }
  // Perform the lookup.
  return _table[code];
}

inline
void
LookupTable::
clear()
{
  _table.clear();
  _shift = 0;
  _first = std::numeric_limits<Code>::max();
}

} // namespace particle
}
