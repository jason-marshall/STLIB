// -*- C++ -*-

#include "stlib/container/ArrayBase.h"

using namespace stlib;

class
  ArrayBaseTester : public container::ArrayBase
{
  //
  // Types.
  //
private:

  typedef container::ArrayBase Base;

public:

  typedef Base::size_type size_type;
  typedef Base::difference_type difference_type;
  typedef Base::Index Index;
  typedef Base::Range Range;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  ArrayBaseTester(const size_type extent, const Index base,
                  const Index stride) :
    Base(extent, base, stride)
  {
  }

  virtual
  ~ArrayBaseTester()
  {
  }

  using Base::rebuild;

private:

  //! Default constructor not implemented.
  ArrayBaseTester();

  //@}
  //--------------------------------------------------------------------------
  //! \name Sequence.
  //@{
public:

  using Base::empty;
  using Base::size;
  using Base::max_size;

  //@}
  //--------------------------------------------------------------------------
  //! \name Array indexing.
  //@{
public:

  using Base::base;
  using Base::setBase;
  using Base::range;
  using Base::stride;
  using Base::offset;
  using Base::arrayIndex;

  //@}
};

void
test(const ArrayBaseTester::size_type extent,
     const ArrayBaseTester::Index base,
     const ArrayBaseTester::Index stride)
{
  typedef ArrayBaseTester ArrayBaseTester;
  typedef ArrayBaseTester::size_type size_type;
  typedef ArrayBaseTester::Index Index;
  typedef ArrayBaseTester::Range Range;

  const size_type size = extent;

  ArrayBaseTester x = ArrayBaseTester(extent, base, stride);
  // array constructor.
  {
    size_type ext = size;
    ArrayBaseTester y(ext, base, stride);
    assert(x == y);
  }
  // Copy constructor.
  {
    ArrayBaseTester y(x);
    assert(x == y);
  }
  // Assignment operator.
  {
    ArrayBaseTester y(extent, base, stride);
    y = x;
    assert(x == y);
  }

  // Container accessors.
  assert(! x.empty());
  assert(x.size() == size);
  assert(x.max_size() == size);

  // Array indexing.
  for (size_type n = 0; n != size; ++n) {
    Index i = base + n;
    assert(x.arrayIndex(i) == stride * Index(n));
  }
  assert(x.base() == base);
  assert(x.stride() == stride);
  assert(x.range() == Range(x.size(), x.base()));

  // Set the bases.
  {
    Index base = 7;
    x.setBase(base);
    assert(x.base() == base);
  }
  // Rebuild
  {
    size_type extent1 = 1;
    Index base1 = -1;
    Index stride1 = 2;
    x.rebuild(extent1, base1, stride1);
    assert(x.size() == extent1);
    assert(x.base() == base1);
    assert(x.stride() == stride1);
  }
}

int
main()
{
  {
    test(7, 0, 1);
    test(7, -2, 1);
    test(7, 0, 3);
    test(7, 0, 1);
  }
  return 0;
}
